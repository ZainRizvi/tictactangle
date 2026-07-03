//! Self-play game generation: MCTS moves with Dirichlet root noise and a
//! visit-count temperature schedule; every position becomes a training
//! sample labeled with the final outcome.

use crate::game::{action_index, encode, initial_state, FEATS};
use crate::mcts::search;
use crate::net::{Net, Sample};
use crate::rng::Rng;
use tictactwo_engine::{apply, Outcome};

pub const MAX_PLY: u32 = 100;
pub const DIRICHLET_EPS: f32 = 0.25;
pub const DIRICHLET_ALPHA: f32 = 0.35;

/// Draw contempt: draws are labeled slightly negative for BOTH sides so the
/// net learns to fight for wins instead of collapsing into safe shuffles
/// (official rules are heavily drawish, and a 0-labeled draw majority
/// otherwise teaches the value head that nothing matters).
pub const DRAW_Z: f32 = -0.15;

pub struct GameStats {
    pub plies: u32,
    pub winner: u8, // 0 = draw/cutoff
}

pub fn play_game(net: &Net, sims: u32, temp_plies: u32, rng: &mut Rng) -> (Vec<Sample>, GameStats) {
    let mut state = initial_state();
    let mut ply = 0u32;
    let mut winner = 0u8;
    let mut pending: Vec<(Sample, u8)> = Vec::new(); // sample + side to move

    loop {
        let res = search(net, &state, sims, Some((DIRICHLET_EPS, DIRICHLET_ALPHA)), rng);
        let total: u32 = res.visits.iter().map(|&(_, v)| v).sum();
        if total == 0 {
            break; // no legal moves shouldn't happen, but don't spin
        }

        let mut feat = [0.0f32; FEATS];
        encode(&state, &mut feat);
        let actions: Vec<usize> = res.visits.iter().map(|&(m, _)| action_index(m)).collect();
        let pi: Vec<f32> = res.visits.iter().map(|&(_, v)| v as f32 / total as f32).collect();
        pending.push((
            Sample { feat, actions, pi, z: 0.0 },
            state.turn,
        ));

        // move selection: sample by visits early, argmax later
        let mv = if ply < temp_plies {
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
                    break; // dead-heat cutoff → labeled DRAW_Z below
                }
            }
            Outcome::Tie => break,
            Outcome::Win(p) => {
                winner = p;
                break;
            }
        }
    }

    let samples = pending
        .into_iter()
        .map(|(mut s, side)| {
            s.z = if winner == 0 {
                DRAW_Z
            } else if winner == side {
                1.0
            } else {
                -1.0
            };
            s
        })
        .collect();
    (samples, GameStats { plies: ply, winner })
}
