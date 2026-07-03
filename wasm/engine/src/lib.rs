//! Tic Tac Two engine: negamax search over the exact game rules, with a
//! small neural network as the leaf evaluator, compiled to WebAssembly.
//! No imports, no allocator use in the hot path — the JS adapter writes the
//! position into a static buffer and reads a packed move back.
//!
//! Rules: 5x5 board, movable 3x3 grid, 4 pieces per player. First two turns
//! per player are placements inside the grid; then a turn is place, slide
//! the grid one step (grid stays on the board), or move an own piece from
//! anywhere to an empty grid cell. Three own pieces in a row inside the grid
//! wins; a slide lighting up lines for both players at once is a tie.
//! Anti-loop: if your grid slide is undone by the opponent's reply slide,
//! you may not immediately repeat it (one-turn ban; the undo itself and
//! every other move stay legal).

// no_std on wasm keeps the artifact tiny; host builds (the RL trainer links
// this crate natively) use std so the cdylib target compiles everywhere.
#![cfg_attr(target_arch = "wasm32", no_std)]

mod weights;

#[cfg(feature = "rl")]
mod mcts_play;
#[cfg(feature = "rl")]
mod rlnet;

pub const EMPTY: u8 = 0;
pub const PIECES: i32 = 4;
pub const MIN_PLACED: i32 = 2;

#[derive(Clone, Copy)]
pub struct State {
    pub board: [u8; 25],
    pub cr: i8,
    pub cc: i8,
    pub turn: u8, // 1 = X, 2 = O
    // Origin center of the last move iff it was a slide (-1,-1 = none).
    pub ls_r: i8,
    pub ls_c: i8,
    // Anti-loop ban lists as 9-bit masks over grid centers
    // (bit = (r-1)*3 + (c-1)): `bn` = centers the side to move may not slide
    // to; `bn_prev` = the list that applied to the player who just moved
    // (carried so an undo can extend it).
    pub bn: u16,
    pub bn_prev: u16,
}

#[inline]
fn center_bit(r: i8, c: i8) -> u16 {
    1 << ((r - 1) * 3 + (c - 1))
}

#[derive(Clone, Copy, PartialEq)]
pub enum Outcome {
    Undecided,
    Win(u8),
    Tie,
}

/// Packed move: kind << 16 | a << 8 | b.
/// kind 0 = place (a = to), 1 = grid (a = dr+1, b = dc+1), 2 = move (a = from, b = to).
pub type Move = u32;

pub const NO_MOVE: Move = 0xFFFF_FFFF;

#[inline]
fn pack(kind: u32, a: u32, b: u32) -> Move {
    (kind << 16) | (a << 8) | b
}

#[inline]
fn other(p: u8) -> u8 {
    3 - p
}

#[inline]
fn in_grid(s: &State, r: i8, c: i8) -> bool {
    (r - s.cr).abs() <= 1 && (c - s.cc).abs() <= 1
}

fn placed(s: &State, p: u8) -> i32 {
    // Pieces never leave the board, so on-board count == total ever placed.
    s.board.iter().filter(|&&v| v == p).count() as i32
}

fn actions_unlocked(s: &State) -> bool {
    placed(s, 1) >= MIN_PLACED && placed(s, 2) >= MIN_PLACED
}

const DIRS: [(i8, i8); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
];

/// The 8 winning lines of the 3x3 grid as center offsets.
const LINES: [[(i8, i8); 3]; 8] = [
    [(-1, -1), (-1, 0), (-1, 1)],
    [(0, -1), (0, 0), (0, 1)],
    [(1, -1), (1, 0), (1, 1)],
    [(-1, -1), (0, -1), (1, -1)],
    [(-1, 0), (0, 0), (1, 0)],
    [(-1, 1), (0, 1), (1, 1)],
    [(-1, -1), (0, 0), (1, 1)],
    [(-1, 1), (0, 0), (1, -1)],
];

fn has_line(s: &State, p: u8) -> bool {
    LINES.iter().any(|line| {
        line.iter().all(|&(dr, dc)| {
            let r = s.cr + dr;
            let c = s.cc + dc;
            s.board[(r * 5 + c) as usize] == p
        })
    })
}

fn outcome(s: &State) -> Outcome {
    let x = has_line(s, 1);
    let o = has_line(s, 2);
    match (x, o) {
        (true, true) => Outcome::Tie,
        (true, false) => Outcome::Win(1),
        (false, true) => Outcome::Win(2),
        (false, false) => Outcome::Undecided,
    }
}

/// Fills `out` with legal moves for the side to move; returns the count.
/// Assumes the game is not already decided.
pub fn legal_moves(s: &State, out: &mut [Move; 96]) -> usize {
    let mut n = 0;
    let unlocked = actions_unlocked(s);
    let reserve = PIECES - placed(s, s.turn);

    let mut empties = [0u32; 9];
    let mut ne = 0;
    for dr in -1..=1i8 {
        for dc in -1..=1i8 {
            let i = ((s.cr + dr) * 5 + (s.cc + dc)) as u32;
            if s.board[i as usize] == EMPTY {
                empties[ne] = i;
                ne += 1;
            }
        }
    }

    if reserve > 0 {
        for &to in &empties[..ne] {
            out[n] = pack(0, to, 0);
            n += 1;
        }
    }

    if unlocked {
        for &(dr, dc) in &DIRS {
            let nr = s.cr + dr;
            let nc = s.cc + dc;
            if (1..=3).contains(&nr) && (1..=3).contains(&nc) && s.bn & center_bit(nr, nc) == 0 {
                out[n] = pack(1, (dr + 1) as u32, (dc + 1) as u32);
                n += 1;
            }
        }
        for from in 0..25u32 {
            if s.board[from as usize] != s.turn {
                continue;
            }
            for &to in &empties[..ne] {
                out[n] = pack(2, from, to);
                n += 1;
            }
        }
    }
    n
}

/// Applies a legal move; returns the new state (turn switched only if the
/// game continues) and the outcome after the move.
pub fn apply(s: &State, m: Move) -> (State, Outcome) {
    let mut n = *s;
    let kind = m >> 16;
    let a = (m >> 8) & 0xFF;
    let b = m & 0xFF;
    // Anti-loop bookkeeping. A slide landing on the previous slide's origin
    // is necessarily an exact undo (slides are one step and this slide
    // starts where the previous one ended): it EXTENDS the opponent's
    // accumulated ban list (carried in bn_prev) with the center their undone
    // slide had created. Any non-undo move clears the opponent's list.
    n.ls_r = -1;
    n.ls_c = -1;
    n.bn = 0;
    n.bn_prev = s.bn;
    match kind {
        0 => n.board[a as usize] = n.turn,
        1 => {
            n.cr += a as i8 - 1;
            n.cc += b as i8 - 1;
            if n.cr == s.ls_r && n.cc == s.ls_c {
                n.bn = s.bn_prev | center_bit(s.cr, s.cc);
            }
            n.ls_r = s.cr;
            n.ls_c = s.cc;
        }
        _ => {
            n.board[a as usize] = EMPTY;
            n.board[b as usize] = n.turn;
        }
    }
    let out = outcome(&n);
    if out == Outcome::Undecided {
        n.turn = other(n.turn);
    }
    (n, out)
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Handcrafted evaluation from the side-to-move perspective, in [-1, 1]-ish.
/// Used for differential testing and as a component check; the shipped
/// evaluator is the neural net below when weights are present.
pub fn eval_handcrafted(s: &State) -> f32 {
    let me = s.turn;
    let opp = other(me);
    let mut score = 0.0f32;
    for line in &LINES {
        let mut mine = 0;
        let mut theirs = 0;
        for &(dr, dc) in line {
            let v = s.board[((s.cr + dr) * 5 + s.cc + dc) as usize];
            if v == me {
                mine += 1;
            } else if v == opp {
                theirs += 1;
            }
        }
        if theirs == 0 {
            score += match mine { 2 => 0.08, 1 => 0.01, _ => 0.0 };
        }
        if mine == 0 {
            score -= match theirs { 2 => 0.08, 1 => 0.01, _ => 0.0 };
        }
    }
    let center = s.board[(s.cr * 5 + s.cc) as usize];
    if center == me {
        score += 0.02;
    } else if center == opp {
        score -= 0.02;
    }
    for i in 0..25usize {
        let v = s.board[i];
        if v == EMPTY {
            continue;
        }
        let lit = in_grid(s, (i / 5) as i8, (i % 5) as i8);
        if !lit {
            score += if v == me { -0.006 } else { 0.006 };
        }
    }
    score
}

/// Encodes the position into the 61 input features the net was trained on,
/// always from the side-to-move perspective. Must match tools/train.mjs.
fn encode(s: &State, feat: &mut [f32; 61]) {
    let me = s.turn;
    let opp = other(me);
    for f in feat.iter_mut() {
        *f = 0.0;
    }
    for i in 0..25usize {
        if s.board[i] == me {
            feat[i] = 1.0;
        } else if s.board[i] == opp {
            feat[25 + i] = 1.0;
        }
    }
    feat[50 + ((s.cr - 1) * 3 + (s.cc - 1)) as usize] = 1.0;
    feat[59] = (PIECES - placed(s, me)) as f32 / PIECES as f32;
    feat[60] = (PIECES - placed(s, opp)) as f32 / PIECES as f32;
}

/// Small MLP value head: 61 -> H1 relu -> H2 relu -> 1 softsign.
/// Layer sizes come from the generated weights module, so difficulty
/// variants can ship differently sized nets from the same source.
fn eval_nn(s: &State) -> f32 {
    use weights::*;
    let mut feat = [0.0f32; 61];
    encode(s, &mut feat);

    // Layer 1 exploits input sparsity: at most ~12 of the 61 features are
    // nonzero (pieces + center one-hot + reserves), and piece planes are 1.0.
    let mut h1 = [0.0f32; H1];
    h1.copy_from_slice(&B1);
    for (i, &f) in feat.iter().enumerate() {
        if f == 0.0 {
            continue;
        }
        let row = &W1[i * H1..(i + 1) * H1];
        if f == 1.0 {
            for j in 0..H1 {
                h1[j] += row[j];
            }
        } else {
            for j in 0..H1 {
                h1[j] += row[j] * f;
            }
        }
    }
    for h in h1.iter_mut() {
        if *h < 0.0 {
            *h = 0.0;
        }
    }
    // Layer 2 skips rows for ReLU-zeroed activations (~half of them).
    let mut h2 = [0.0f32; H2];
    h2.copy_from_slice(&B2);
    for (i, &v) in h1.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let row = &W2[i * H2..(i + 1) * H2];
        for j in 0..H2 {
            h2[j] += row[j] * v;
        }
    }
    for h in h2.iter_mut() {
        if *h < 0.0 {
            *h = 0.0;
        }
    }
    let mut out = B3;
    for (i, &v) in h2.iter().enumerate() {
        out += W3[i] * v;
    }
    // cheap tanh approximation, monotone and bounded
    out / (1.0 + if out < 0.0 { -out } else { out })
}

#[inline]
fn evaluate(s: &State) -> f32 {
    if weights::TRAINED {
        eval_nn(s)
    } else {
        eval_handcrafted(s)
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

pub const WIN: f32 = 1000.0;

struct Ctx {
    nodes: i64,
    rng: u64,
}

impl Ctx {
    fn rand(&mut self) -> u32 {
        // xorshift64*
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 33) as u32
    }
}

/// Negamax with alpha-beta. Returns the score from `s.turn`'s perspective.
/// The node budget counts every visited position including leaves, since
/// leaf NN evaluations dominate the cost.
fn search(s: &State, depth: i32, mut alpha: f32, beta: f32, ctx: &mut Ctx) -> f32 {
    ctx.nodes -= 1;
    if depth == 0 || ctx.nodes <= 0 {
        return evaluate(s);
    }

    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    let mut best = -WIN * 2.0;

    for &m in &moves[..n] {
        let (ns, out) = apply(s, m);
        let v = match out {
            Outcome::Win(p) if p == s.turn => WIN + depth as f32,
            Outcome::Win(_) => -(WIN + depth as f32), // our slide lit their line
            Outcome::Tie => 0.0,
            Outcome::Undecided => -search(&ns, depth - 1, -beta, -alpha, ctx),
        };
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Iterative-deepening root search; returns the chosen move.
pub fn choose(s: &State, max_depth: i32, node_budget: i64, seed: u64) -> Move {
    choose_scored(s, max_depth, node_budget, seed).0
}

/// Like `choose`, also returning the best score found in the last trusted
/// round (>= WIN means a forced win was proven within the search depth).
pub fn choose_scored(s: &State, max_depth: i32, node_budget: i64, seed: u64) -> (Move, f32) {
    let mut ctx = Ctx {
        nodes: node_budget,
        rng: seed | 1,
    };
    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    if n == 0 {
        return (NO_MOVE, 0.0);
    }

    // Random tiebreak ordering so equal positions don't repeat forever.
    for i in (1..n).rev() {
        let j = (ctx.rand() as usize) % (i + 1);
        moves.swap(i, j);
    }

    let mut best_move = moves[0];
    let mut best_score = 0.0f32;
    let mut scores = [0.0f32; 96];
    for depth in 1..=max_depth {
        let mut round_best = -WIN * 4.0;
        let mut round_move = NO_MOVE;
        let mut alpha = -WIN * 4.0;
        let start_nodes = ctx.nodes;

        for k in 0..n {
            let m = moves[k];
            let (ns, out) = apply(s, m);
            let v = match out {
                Outcome::Win(p) if p == s.turn => WIN + depth as f32,
                Outcome::Win(_) => -(WIN + depth as f32),
                Outcome::Tie => 0.0,
                Outcome::Undecided => -search(&ns, depth - 1, -(WIN * 4.0), -alpha, &mut ctx),
            };
            scores[k] = v;
            if v > round_best {
                round_best = v;
                round_move = m;
            }
            if v > alpha {
                alpha = v;
            }
        }

        let exhausted = ctx.nodes <= 0;
        if round_move != NO_MOVE && (!exhausted || round_best >= WIN) {
            best_move = round_move;
            best_score = round_best;
        }
        if round_best >= WIN || exhausted {
            break;
        }

        // Order the next round by this round's scores (insertion sort keeps
        // moves and scores paired) — much tighter alpha-beta pruning.
        for a in 1..n {
            let (mv, sc) = (moves[a], scores[a]);
            let mut b = a;
            while b > 0 && scores[b - 1] < sc {
                moves[b] = moves[b - 1];
                scores[b] = scores[b - 1];
                b -= 1;
            }
            moves[b] = mv;
            scores[b] = sc;
        }
        // Stop deepening when the next depth can't plausibly fit the
        // remaining budget (alpha-beta grows roughly ~6-8x per ply here).
        let used = start_nodes - ctx.nodes;
        if ctx.nodes < used * 6 {
            break;
        }
    }
    (best_move, best_score)
}

// ---------------------------------------------------------------------------
// WASM API
// ---------------------------------------------------------------------------

static mut IN_BUF: [u8; 40] = [0; 40];
static mut SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Pointer to the 40-byte input buffer:
/// bytes 0..25 board (0/1/2), 25 center row, 26 center col, 27 side to move,
/// 28/29 last slide origin (0 = none), 30/31 banned-centers mask (u16 LE),
/// 32/33 previous mover's banned-centers mask (u16 LE), 34..40 reserved.
#[no_mangle]
pub extern "C" fn input_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(IN_BUF) as *mut u8
}

#[no_mangle]
pub extern "C" fn set_seed(seed: u32) {
    unsafe { SEED = (seed as u64) << 16 | 0x9E37 }
}

/// Reads and validates the position from the input buffer. Board cells must
/// be 0/1/2 with at most 4 pieces per side (this also bounds legal_moves
/// well under its output buffer) — garbage is rejected rather than risking
/// out-of-bounds panics.
fn read_input() -> Option<State> {
    let buf = unsafe { &*core::ptr::addr_of!(IN_BUF) };
    let mut s = State {
        board: [0; 25],
        cr: buf[25] as i8,
        cc: buf[26] as i8,
        turn: buf[27],
        ls_r: buf[28] as i8,
        ls_c: buf[29] as i8,
        bn: u16::from_le_bytes([buf[30], buf[31]]),
        bn_prev: u16::from_le_bytes([buf[32], buf[33]]),
    };
    s.board.copy_from_slice(&buf[..25]);
    if !(1..=3).contains(&s.cr) || !(1..=3).contains(&s.cc) || !(1..=2).contains(&s.turn) {
        return None;
    }
    // Last-slide origin: (0,0) means none; anything else must be a valid
    // grid center. Ban masks may only use the 9 center bits.
    if s.ls_r == 0 && s.ls_c == 0 {
        s.ls_r = -1;
        s.ls_c = -1;
    } else if !(1..=3).contains(&s.ls_r) || !(1..=3).contains(&s.ls_c) {
        return None;
    }
    if s.bn & !0x1FF != 0 || s.bn_prev & !0x1FF != 0 {
        return None;
    }
    if s.board.iter().any(|&v| v > 2) || placed(&s, 1) > PIECES || placed(&s, 2) > PIECES {
        return None;
    }
    Some(s)
}

fn next_seed() -> u64 {
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        SEED
    }
}

#[no_mangle]
pub extern "C" fn choose_move(max_depth: u32, node_budget: u32) -> u32 {
    let Some(s) = read_input() else {
        return NO_MOVE;
    };
    choose(&s, max_depth.min(8) as i32, node_budget as i64, next_seed())
}

/// AlphaZero-style play: MCTS over the embedded RL policy/value net.
/// Available in `rl`-feature builds (the "impossible" difficulty).
#[cfg(feature = "rl")]
#[no_mangle]
pub extern "C" fn choose_move_mcts(sims: u32) -> u32 {
    let Some(s) = read_input() else {
        return NO_MOVE;
    };
    mcts_play::choose_mcts(&s, sims.clamp(16, 20_000), next_seed())
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
