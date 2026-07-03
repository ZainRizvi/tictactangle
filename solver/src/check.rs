//! Post-solve correctness checks against a solved table.
//!
//! (a) Known hand positions mirrored from test/wasm-engine.test.mjs and
//!     test/game.test.mjs, with asserted game values / winning move types.
//! (b) A randomized cross-check against the engine's alpha-beta search: where
//!     alpha-beta PROVES a forced win (score >= WIN) for the side to move, the
//!     solver must label that position WIN.

use crate::canon::Canon;
use crate::solve::{optimal_moves, Solver, LOSS, UNKNOWN, WIN};
use tictactwo_engine::{apply, choose_scored, legal_moves, Outcome, State, WIN as AB_WIN};

const X: u8 = 1;
const O: u8 = 2;
#[inline]
fn idx(r: usize, c: usize) -> usize {
    r * 5 + c
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Build a board-only state with no anti-loop slide history (turn set by caller).
fn st(board: [u8; 25], cr: i8, cc: i8, turn: u8) -> State {
    State {
        board,
        cr,
        cc,
        turn,
        ls_r: -1,
        ls_c: -1,
        bn_r: -1,
        bn_c: -1,
    }
}

pub fn run(table: &str) {
    println!("=== solved-table correctness checks ===\n");
    let solver = Solver::load(table).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {} — run `solve` first", table, e);
        std::process::exit(1);
    });

    let mut fails = 0;

    // ---- (a) Known hand positions -------------------------------------------

    // mate-in-1 by placement: X at (1,1),(1,2); O at (3,1),(3,2). X to move,
    // 2 reserve each. X places (1,3) to complete the grid's top row.
    // (test/wasm-engine.test.mjs "takes an immediate win")
    {
        let mut b = [0u8; 25];
        b[idx(1, 1)] = X;
        b[idx(1, 2)] = X;
        b[idx(3, 1)] = O;
        b[idx(3, 2)] = O;
        let s = st(b, 2, 2, X);
        fails += expect_win(&solver, &s, "mate-in-1 placement (X)");
        // and the completing place move should be among the optimal moves
        fails += expect_move_kind(&solver, &s, 0, "placement win move kind");
    }

    // blocks an immediate loss: X threatens; O to move must prevent X's win.
    // O has no win of its own, so this is at best a draw / at worst loss — but
    // O should NOT be a forced loss just from being on defense here; we only
    // assert it's not WIN for O (O can't force a win in one move here). Weaker
    // check: value is not WIN for the side to move (O).
    {
        let mut b = [0u8; 25];
        b[idx(1, 1)] = X;
        b[idx(1, 2)] = X;
        b[idx(3, 1)] = O;
        b[idx(2, 3)] = O;
        let s = st(b, 2, 2, O);
        let c = Canon::from_state(&s).unwrap();
        let v = solver.value_of(&c);
        if v == WIN {
            eprintln!("  FAIL: defender O labeled WIN in 'blocks an immediate loss'");
            fails += 1;
        } else {
            println!("  OK: defender O not a forced win ({})", lbl(v));
        }
        // Whatever O's value, an optimal move must not hand X an immediate win.
        let (_v, best) = optimal_moves(&s, &solver);
        let mut safe = false;
        for &m in &best {
            let (ns, out) = apply(&s, m);
            if out == Outcome::Undecided {
                // X must not have an immediate win from ns
                let mut mv = [0u32; 96];
                let n = legal_moves(&ns, &mut mv);
                let x_wins = mv[..n].iter().any(|&mm| {
                    matches!(apply(&ns, mm).1, Outcome::Win(p) if p == X)
                });
                if !x_wins {
                    safe = true;
                }
            } else if matches!(out, Outcome::Win(p) if p == O) || out == Outcome::Tie {
                safe = true;
            }
        }
        if safe {
            println!("  OK: O has a defensive optimal move");
        } else {
            eprintln!("  FAIL: no safe optimal move for O in block scenario");
            fails += 1;
        }
    }

    // grid-slide win: X row on board row 3 cols 1-3 + a 4th X; grid at (1,2).
    // Only sliding down to (2,2) lights the row. X to move, no reserve.
    // (test/wasm-engine.test.mjs "wins with a grid slide")
    {
        let mut b = [0u8; 25];
        b[idx(3, 1)] = X;
        b[idx(3, 2)] = X;
        b[idx(3, 3)] = X;
        b[idx(0, 0)] = X;
        b[idx(0, 2)] = O;
        b[idx(0, 3)] = O;
        b[idx(4, 0)] = O;
        b[idx(4, 4)] = O;
        let s = st(b, 1, 2, X);
        fails += expect_win(&solver, &s, "mate-in-1 grid-slide (X)");
        fails += expect_move_kind(&solver, &s, 1, "grid-slide win move kind");
    }

    // piece-move win: X two on grid row 2 with the third cell empty; no reserve.
    // (test/wasm-engine.test.mjs "wins with a piece move")
    {
        let mut b = [0u8; 25];
        b[idx(2, 1)] = X;
        b[idx(2, 2)] = X;
        b[idx(4, 4)] = X;
        b[idx(0, 0)] = X;
        b[idx(1, 1)] = O;
        b[idx(3, 3)] = O;
        b[idx(0, 4)] = O;
        b[idx(4, 0)] = O;
        let s = st(b, 2, 2, X);
        fails += expect_win(&solver, &s, "mate-in-1 piece-move (X)");
        fails += expect_move_kind(&solver, &s, 2, "piece-move win move kind");
    }

    // tie slide: X row board row 2 cols 2-4, O row board row 4 cols 2-4, grid
    // at (2,2); sliding to (3,3) lights both -> tie. This is a to-move position
    // where the ONLY value-relevant fact we assert is that the mover can secure
    // at least a draw (value != LOSS), since a tie move is available.
    // (test/game.test.mjs "grid slide revealing lines for both players is a tie")
    {
        let mut b = [0u8; 25];
        b[idx(2, 2)] = X;
        b[idx(2, 3)] = X;
        b[idx(2, 4)] = X;
        b[idx(4, 2)] = O;
        b[idx(4, 3)] = O;
        b[idx(4, 4)] = O;
        // X to move (3 placed each; reserve 1 each). Counts (3,3): (a,b)=(3,3).
        let s = st(b, 2, 2, X);
        let c = Canon::from_state(&s).unwrap();
        let v = solver.value_of(&c);
        if v == LOSS {
            eprintln!("  FAIL: tie-slide position labeled LOSS (a tie is available)");
            fails += 1;
        } else {
            println!("  OK: tie-slide position not LOSS ({})", lbl(v));
        }
    }

    // self-loss slide: O row on board row 3 cols 1-3, grid at (1,2). If X (to
    // move) is FORCED to reveal it, X loses; but here X has other moves, so we
    // only assert the position is well-defined and X won't voluntarily slide
    // onto O's line as an "optimal" move unless it's the best value.
    // (test/game.test.mjs "sliding the grid onto only the opponent line loses")
    {
        let mut b = [0u8; 25];
        b[idx(3, 1)] = O;
        b[idx(3, 2)] = O;
        b[idx(3, 3)] = O;
        b[idx(0, 1)] = X;
        b[idx(0, 2)] = X;
        b[idx(1, 1)] = X;
        let s = st(b, 1, 2, X);
        let c = Canon::from_state(&s).unwrap();
        let v = solver.value_of(&c);
        // The revealing slide (dr=1,dc=0) must never be an optimal move unless
        // every move loses (then value is LOSS and all moves are "equal-worst").
        let (_v, best) = optimal_moves(&s, &solver);
        let reveal = (1u32 << 16) | ((1u32 + 1) << 8) | (0u32 + 1); // kind1 dr=1 dc=0
        let picks_reveal = best.contains(&reveal);
        if v != LOSS && picks_reveal {
            eprintln!("  FAIL: self-loss slide chosen as optimal despite non-LOSS value");
            fails += 1;
        } else {
            println!("  OK: self-loss slide not mis-chosen ({})", lbl(v));
        }
    }

    // ---- (b) Alpha-beta cross-check -----------------------------------------
    println!("\nAlpha-beta cross-check (proven AB wins must be solver WIN)...");
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let mut proven = 0;
    let mut checked = 0;
    let mut ab_disagree = 0;
    for _ in 0..4000 {
        let s = random_reachable(&mut rng);
        let Some(c) = Canon::from_state(&s) else {
            continue;
        };
        // Skip already-decided positions.
        if crate::solve::has_any_line(&s) {
            continue;
        }
        checked += 1;
        // Deep alpha-beta with a generous budget; score >= AB_WIN proves a
        // forced win for the side to move within the searched depth.
        let (_m, score) = choose_scored(&s, 8, 5_000_000, 0x1234 ^ rng.next());
        if score >= AB_WIN {
            proven += 1;
            let v = solver.value_of(&c);
            if v != WIN {
                ab_disagree += 1;
                if ab_disagree <= 10 {
                    eprintln!(
                        "  FAIL: AB proves win (score {:.1}) but solver says {} for {}",
                        score,
                        lbl(v),
                        board_str(&s)
                    );
                }
            }
        }
    }
    println!(
        "  checked {} positions, {} AB-proven wins, {} disagreements",
        checked, proven, ab_disagree
    );
    fails += ab_disagree;

    // Reverse direction (advisory): where the solver says LOSS, alpha-beta must
    // never prove a win for the same side. (A solver LOSS means the opponent
    // forces a win, so the side to move cannot have a proven win.)
    let mut loss_disagree = 0;
    let mut rng = Rng(0x9999_1111_2222_3333);
    for _ in 0..4000 {
        let s = random_reachable(&mut rng);
        let Some(c) = Canon::from_state(&s) else {
            continue;
        };
        if crate::solve::has_any_line(&s) {
            continue;
        }
        if solver.value_of(&c) == LOSS {
            let (_m, score) = choose_scored(&s, 8, 5_000_000, 0x55 ^ rng.next());
            if score >= AB_WIN {
                loss_disagree += 1;
                if loss_disagree <= 10 {
                    eprintln!(
                        "  FAIL: solver says LOSS but AB proves a win for side to move: {}",
                        board_str(&s)
                    );
                }
            }
        }
    }
    println!("  solver-LOSS vs AB-proven-win disagreements: {}", loss_disagree);
    fails += loss_disagree;

    println!(
        "\n=== checks {} ===",
        if fails == 0 {
            "PASSED".to_string()
        } else {
            format!("FAILED with {} problems", fails)
        }
    );
    if fails != 0 {
        std::process::exit(1);
    }
}

/// Assert the side to move is labeled WIN.
fn expect_win(solver: &Solver, s: &State, name: &str) -> u64 {
    let c = Canon::from_state(s).expect("valid position");
    let v = solver.value_of(&c);
    if v == WIN {
        println!("  OK: {} = WIN", name);
        0
    } else {
        eprintln!("  FAIL: {} expected WIN, got {}", name, lbl(v));
        1
    }
}

/// Assert some optimal move has the given kind (0 place, 1 grid, 2 move).
fn expect_move_kind(solver: &Solver, s: &State, kind: u32, name: &str) -> u64 {
    let (_v, best) = optimal_moves(s, solver);
    if best.iter().any(|&m| (m >> 16) == kind) {
        println!("  OK: {} present", name);
        0
    } else {
        eprintln!("  FAIL: {} — no optimal move of kind {}", name, kind);
        1
    }
}

fn lbl(v: u8) -> &'static str {
    match v {
        WIN => "WIN",
        LOSS => "LOSS",
        UNKNOWN => "DRAW",
        _ => "?",
    }
}

fn board_str(s: &State) -> String {
    let mut out = String::with_capacity(40);
    for i in 0..25 {
        out.push((b'0' + s.board[i]) as char);
    }
    format!("{} c=({},{}) t={}", out, s.cr, s.cc, s.turn)
}

/// Generate a random *reachable* to-move position: pick counts from a valid
/// (a,b) block, scatter that many pieces on distinct squares, random center.
/// May still be a decided position; the caller filters those out.
fn random_reachable(rng: &mut Rng) -> State {
    use crate::index::BLOCKS;
    let (a, b) = BLOCKS[rng.below(BLOCKS.len())];
    let mut board = [0u8; 25];
    let mut placed = 0u32;
    let target = a + b;
    while placed < target {
        let sq = rng.below(25);
        if board[sq] == 0 {
            board[sq] = if placed < a { X } else { O };
            placed += 1;
        }
    }
    let cr = 1 + rng.below(3) as i8;
    let cc = 1 + rng.below(3) as i8;
    st(board, cr, cc, X)
}
