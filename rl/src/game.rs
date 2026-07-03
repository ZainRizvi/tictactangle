//! Bridging helpers over the engine's rules: feature encoding for the net
//! and a fixed indexing of the full action space for the policy head.

use tictactwo_engine::{legal_moves, Move, State};

pub const FEATS: usize = 61;

/// Action space: 25 placements + 8 slides + 625 (from,to) piece moves.
pub const ACTIONS: usize = 25 + 8 + 625;

/// Encode a position from the side-to-move perspective.
/// Layout matches the supervised pipeline: own 0..25, opp 25..50,
/// grid-center one-hot 50..59, reserves 59/60.
pub fn encode(s: &State, out: &mut [f32; FEATS]) {
    out.fill(0.0);
    let me = s.turn;
    let opp = 3 - me;
    let mut mine = 0i32;
    let mut theirs = 0i32;
    for i in 0..25 {
        if s.board[i] == me {
            out[i] = 1.0;
            mine += 1;
        } else if s.board[i] == opp {
            out[25 + i] = 1.0;
            theirs += 1;
        }
    }
    out[50 + ((s.cr - 1) * 3 + (s.cc - 1)) as usize] = 1.0;
    out[59] = (4 - mine) as f32 / 4.0;
    out[60] = (4 - theirs) as f32 / 4.0;
}

const DIR_LIST: [(i8, i8); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
];

/// Packed engine move -> policy index.
pub fn action_index(m: Move) -> usize {
    let kind = m >> 16;
    let a = ((m >> 8) & 0xFF) as usize;
    let b = (m & 0xFF) as usize;
    match kind {
        0 => a,                                   // place at cell a
        1 => {
            let dr = a as i8 - 1;
            let dc = b as i8 - 1;
            let d = DIR_LIST.iter().position(|&(r, c)| r == dr && c == dc).unwrap();
            25 + d
        }
        _ => 33 + a * 25 + b,                     // move from a to b
    }
}

/// Legal moves plus their policy indices.
pub fn legal_with_indices(s: &State) -> ([Move; 96], [usize; 96], usize) {
    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    let mut idxs = [0usize; 96];
    for k in 0..n {
        idxs[k] = action_index(moves[k]);
    }
    (moves, idxs, n)
}

pub fn initial_state() -> State {
    State {
        board: [0; 25],
        cr: 2,
        cc: 2,
        turn: 1,
        ls_r: -1,
        ls_c: -1,
        bn_r: -1,
        bn_c: -1,
    }
}
