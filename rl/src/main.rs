//! AlphaZero-style training for Tic Tac Two (official rules).
//!
//!   cargo run --release -- loop  [iters] [games/iter] [sims] [threads]
//!   cargo run --release -- eval  <a.bin> <b.bin> [games] [sims]
//!   cargo run --release -- vsab  <model.bin> [games] [sims] [depth] [budget]
//!
//! `loop` self-plays, trains, gates against the current best, and writes
//! models under rl/models/.

mod game;
mod mcts;
mod net;
mod rng;
mod selfplay;

use game::{initial_state, FEATS};
use net::{Net, Sample, Trainer};
use rng::Rng;
use selfplay::{play_game, MAX_PLY};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tictactwo_engine::{apply, choose, choose_scored, Outcome, WIN};

const BATCH: usize = 128;
const BUFFER_CAP: usize = 80_000;
const TEMP_PLIES: u32 = 12;
const GATE_GAMES: u32 = 40;
const GATE_THRESHOLD: f32 = 0.55;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("loop");
    match cmd {
        "loop" => run_loop(
            arg(&args, 2, 30),
            arg(&args, 3, 240),
            arg(&args, 4, 160),
            arg(&args, 5, 8),
        ),
        "eval" => {
            let a = Net::load(Path::new(&args[2])).expect("load a");
            let b = Net::load(Path::new(&args[3])).expect("load b");
            let (sa, det) = match_nets(&a, &b, arg(&args, 4, 40), arg(&args, 5, 200), 777);
            println!("A score {:.1}% over {} decisive-weighted games", sa * 100.0, det);
        }
        "vsab" => vs_alphabeta(
            &args[2],
            arg(&args, 3, 20),
            arg(&args, 4, 400),
            arg(&args, 5, 8),
            arg(&args, 6, 600_000),
        ),
        other => eprintln!("unknown command: {other}"),
    }
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn models_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    std::fs::create_dir_all(&d).ok();
    d
}

// ---------------------------------------------------------------------------

fn run_loop(iters: u32, games_per_iter: u32, sims: u32, threads: u32) {
    let dir = models_dir();
    let best_path = dir.join("best.bin");
    let mut rng = Rng::new(0xC0FFEE);
    let mut best = if best_path.exists() {
        println!("resuming from {}", best_path.display());
        Net::load(&best_path).expect("load best")
    } else {
        let n = Net::new(&mut rng);
        n.save(&best_path).unwrap();
        n
    };

    let mut buffer: VecDeque<Sample> = VecDeque::new();
    let mut candidate = best.clone_net();
    let mut trainer = Trainer::new(&mut candidate, 1e-3);

    for iter in 1..=iters {
        let t0 = std::time::Instant::now();

        // ---- self-play with the current best ----
        let shared = Arc::new(best.clone_net());
        let per_thread = games_per_iter.div_ceil(threads);
        let mut all: Vec<Sample> = Vec::new();
        let mut plies_sum = 0u64;
        let mut decisive = 0u32;
        let mut x_wins = 0u32;
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for t in 0..threads {
                let net = Arc::clone(&shared);
                handles.push(scope.spawn(move || {
                    let mut trng = Rng::new(0x9E3779B9 * (iter as u64 + 1) + t as u64 * 7919);
                    let mut samples = Vec::new();
                    let mut stats = (0u64, 0u32, 0u32); // plies, decisive, x wins
                    for _ in 0..per_thread {
                        let (s, gs) = play_game(&net, sims, TEMP_PLIES, &mut trng);
                        samples.extend(s);
                        stats.0 += gs.plies as u64;
                        if gs.winner != 0 {
                            stats.1 += 1;
                            if gs.winner == 1 {
                                stats.2 += 1;
                            }
                        }
                    }
                    (samples, stats)
                }));
            }
            for h in handles {
                let (s, st) = h.join().unwrap();
                all.extend(s);
                plies_sum += st.0;
                decisive += st.1;
                x_wins += st.2;
            }
        });
        let games = per_thread * threads;
        let n_new = all.len();
        for s in all {
            if buffer.len() >= BUFFER_CAP {
                buffer.pop_front();
            }
            buffer.push_back(s);
        }

        // ---- train the candidate on the buffer ----
        let steps = (buffer.len() / BATCH).clamp(50, 600);
        let mut vloss_sum = 0.0;
        let mut ploss_sum = 0.0;
        let samples: Vec<&Sample> = buffer.iter().collect();
        for _ in 0..steps {
            let mut bv2_grad = 0.0;
            let mut vl = 0.0;
            let mut pl = 0.0;
            for _ in 0..BATCH {
                let s = samples[rng.below(samples.len())];
                let (a, b) = trainer.accumulate(&candidate, s, &mut bv2_grad, 1.0 / BATCH as f32);
                vl += a;
                pl += b;
            }
            trainer.step(&mut candidate, bv2_grad);
            vloss_sum += vl / BATCH as f32;
            ploss_sum += pl / BATCH as f32;
        }

        // ---- gate candidate vs best ----
        let (score, _) = match_nets(&candidate, &best, GATE_GAMES, sims, 1234 + iter as u64);
        let promoted = score >= GATE_THRESHOLD;
        if promoted {
            best = candidate.clone_net();
            best.save(&best_path).unwrap();
        }
        best.save(&dir.join(format!("gen-{iter}.bin"))).unwrap();

        println!(
            "iter {iter}: {games} games, {n_new} samples (buf {}), avg len {:.1}, decisive {:.0}% (X {:.0}%), vloss {:.3}, ploss {:.3}, gate {:.0}% {} [{:.1}s]",
            buffer.len(),
            plies_sum as f32 / games as f32,
            decisive as f32 * 100.0 / games as f32,
            if decisive > 0 { x_wins as f32 * 100.0 / decisive as f32 } else { 0.0 },
            vloss_sum / steps as f32,
            ploss_sum / steps as f32,
            score * 100.0,
            if promoted { "PROMOTED" } else { "kept old" },
            t0.elapsed().as_secs_f32(),
        );
    }
    println!("done; best model at {}", best_path.display());
}

// ---------------------------------------------------------------------------

/// Plays `games` between two nets (alternating colors, light opening
/// randomness). Returns (score for A in [0,1] counting draws 0.5, games).
fn match_nets(a: &Net, b: &Net, games: u32, sims: u32, seed: u64) -> (f32, u32) {
    let mut score = 0.0;
    let mut rng = Rng::new(seed);
    for g in 0..games {
        let a_is_x = g % 2 == 0;
        let winner = play_match_game(a, b, a_is_x, sims, &mut rng);
        score += match winner {
            0 => 0.5,
            1 => {
                if a_is_x {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                if a_is_x {
                    0.0
                } else {
                    1.0
                }
            }
        };
    }
    (score / games as f32, games)
}

fn play_match_game(a: &Net, b: &Net, a_is_x: bool, sims: u32, rng: &mut Rng) -> u8 {
    let mut state = initial_state();
    let mut ply = 0u32;
    loop {
        let net = if (state.turn == 1) == a_is_x { a } else { b };
        // Tiny root noise for opening diversity in the first two plies.
        let noise = if ply < 2 { Some((0.25, 0.5)) } else { None };
        let res = mcts::search(net, &state, sims, noise, rng);
        let mv = if ply < 2 {
            let total: u32 = res.visits.iter().map(|&(_, v)| v).sum();
            let mut r = rng.uniform() * total as f32;
            let mut chosen = res.visits[0].0;
            for &(m, v) in &res.visits {
                r -= v as f32;
                chosen = m;
                if r <= 0.0 {
                    break;
                }
            }
            chosen
        } else {
            res.visits.iter().max_by_key(|&&(_, v)| v).unwrap().0
        };
        let (ns, out) = apply(&state, mv);
        let ns = crate::game::no_ban(ns);
        ply += 1;
        match out {
            Outcome::Undecided => {
                state = ns;
                if ply >= MAX_PLY {
                    return 0;
                }
            }
            Outcome::Tie => return 0,
            Outcome::Win(p) => return p,
        }
    }
}

// ---------------------------------------------------------------------------

/// Reference match: MCTS+net vs the alpha-beta engine (whatever weights the
/// engine crate was last built with).
fn vs_alphabeta(model: &str, games: u32, sims: u32, depth: u32, budget: u32) {
    let net = Net::load(Path::new(model)).expect("load model");
    let mut rng = Rng::new(4242);
    let mut w = 0;
    let mut l = 0;
    let mut d = 0;
    let mut w_as_x = 0;
    let mut w_as_o = 0;
    for g in 0..games {
        let net_is_x = g % 2 == 0;
        let mut state = initial_state();
        let mut ply = 0u32;
        let winner = loop {
            let mv = if (state.turn == 1) == net_is_x {
                // Same hybrid as the shipped impossible engine: take proven
                // forced wins via alpha-beta, play everything else via MCTS.
                let (tactic, score) = choose_scored(&state, 8, 250_000, rng.next_u64());
                if score >= WIN || score >= 0.5 {
                    let (ns, out) = apply(&state, tactic);
                    let ns = crate::game::no_ban(ns);
                    ply += 1;
                    match out {
                        Outcome::Undecided => {
                            state = ns;
                            if ply >= MAX_PLY {
                                break 0;
                            }
                            continue;
                        }
                        Outcome::Tie => break 0,
                        Outcome::Win(p) => break p,
                    }
                }
                let noise = if ply < 2 { Some((0.25, 0.5)) } else { None };
                let res = mcts::search(&net, &state, sims, noise, &mut rng);
                if ply < 2 {
                    let total: u32 = res.visits.iter().map(|&(_, v)| v).sum();
                    let mut r = rng.uniform() * total as f32;
                    let mut chosen = res.visits[0].0;
                    for &(m, v) in &res.visits {
                        r -= v as f32;
                        chosen = m;
                        if r <= 0.0 {
                            break;
                        }
                    }
                    chosen
                } else {
                    res.visits.iter().max_by_key(|&&(_, v)| v).unwrap().0
                }
            } else {
                choose(&state, depth as i32, budget as i64, rng.next_u64())
            };
            let (ns, out) = apply(&state, mv);
        let ns = crate::game::no_ban(ns);
            ply += 1;
            match out {
                Outcome::Undecided => {
                    state = ns;
                    if ply >= MAX_PLY {
                        break 0;
                    }
                }
                Outcome::Tie => break 0,
                Outcome::Win(p) => break p,
            }
        };
        if winner == 0 {
            d += 1;
        } else if (winner == 1) == net_is_x {
            w += 1;
            if net_is_x {
                w_as_x += 1;
            } else {
                w_as_o += 1;
            }
        } else {
            l += 1;
        }
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!(
        "\nnet vs alpha-beta(depth {depth}, budget {budget}): W {w} (X {w_as_x} / O {w_as_o})  L {l}  D {d}"
    );
    let _ = FEATS; // silence unused warnings if any
}
