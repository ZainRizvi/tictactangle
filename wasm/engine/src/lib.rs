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
//! No take-backs: immediately after a slide, the reply may not slide the
//! grid straight back to the position it just left.

#![no_std]

mod weights;

pub const EMPTY: u8 = 0;
pub const PIECES: i32 = 4;
pub const MIN_PLACED: i32 = 2;

#[derive(Clone, Copy)]
pub struct State {
    pub board: [u8; 25],
    pub cr: i8,
    pub cc: i8,
    pub turn: u8, // 1 = X, 2 = O
    // Center a grid slide may not land on this turn (-1,-1 = none).
    pub br: i8,
    pub bc: i8,
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
            if (1..=3).contains(&nr) && (1..=3).contains(&nc) && !(nr == s.br && nc == s.bc) {
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
    // A slide arms the take-back ban for the reply; other moves clear it.
    n.br = -1;
    n.bc = -1;
    match kind {
        0 => n.board[a as usize] = n.turn,
        1 => {
            n.br = n.cr;
            n.bc = n.cc;
            n.cr += a as i8 - 1;
            n.cc += b as i8 - 1;
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

/// Small MLP value head: 61 -> 64 relu -> 32 relu -> 1 tanh-ish.
fn eval_nn(s: &State) -> f32 {
    use weights::*;
    let mut feat = [0.0f32; 61];
    encode(s, &mut feat);

    let mut h1 = [0.0f32; 64];
    for (j, h) in h1.iter_mut().enumerate() {
        let mut acc = B1[j];
        for (i, &f) in feat.iter().enumerate() {
            acc += W1[i * 64 + j] * f;
        }
        *h = if acc > 0.0 { acc } else { 0.0 };
    }
    let mut h2 = [0.0f32; 32];
    for (j, h) in h2.iter_mut().enumerate() {
        let mut acc = B2[j];
        for (i, &v) in h1.iter().enumerate() {
            acc += W2[i * 32 + j] * v;
        }
        *h = if acc > 0.0 { acc } else { 0.0 };
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

const WIN: f32 = 1000.0;

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
fn search(s: &State, depth: i32, mut alpha: f32, beta: f32, ctx: &mut Ctx) -> f32 {
    if depth == 0 || ctx.nodes <= 0 {
        return evaluate(s);
    }
    ctx.nodes -= 1;

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
    let mut ctx = Ctx {
        nodes: node_budget,
        rng: seed | 1,
    };
    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    if n == 0 {
        return NO_MOVE;
    }

    // Random tiebreak ordering so equal positions don't repeat forever.
    for i in (1..n).rev() {
        let j = (ctx.rand() as usize) % (i + 1);
        moves.swap(i, j);
    }

    let mut best_move = moves[0];
    for depth in 1..=max_depth {
        let mut round_best = -WIN * 4.0;
        let mut round_move = NO_MOVE;
        let mut alpha = -WIN * 4.0;
        let start_nodes = ctx.nodes;

        for &m in &moves[..n] {
            let (ns, out) = apply(s, m);
            let v = match out {
                Outcome::Win(p) if p == s.turn => WIN + depth as f32,
                Outcome::Win(_) => -(WIN + depth as f32),
                Outcome::Tie => 0.0,
                Outcome::Undecided => -search(&ns, depth - 1, -(WIN * 4.0), -alpha, &mut ctx),
            };
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
        }
        if round_best >= WIN || exhausted {
            break;
        }
        // If this depth used almost nothing, keep deepening; otherwise stop
        // when the next depth can't plausibly fit the remaining budget.
        let used = start_nodes - ctx.nodes;
        if ctx.nodes < used * 20 {
            break;
        }
    }
    best_move
}

// ---------------------------------------------------------------------------
// WASM API
// ---------------------------------------------------------------------------

static mut IN_BUF: [u8; 32] = [0; 32];
static mut SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Pointer to the 32-byte input buffer:
/// bytes 0..25 board (0/1/2), 25 center row, 26 center col, 27 side to move,
/// 28/29 banned slide center (0 = none).
#[no_mangle]
pub extern "C" fn input_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(IN_BUF) as *mut u8
}

#[no_mangle]
pub extern "C" fn set_seed(seed: u32) {
    unsafe { SEED = (seed as u64) << 16 | 0x9E37 }
}

#[no_mangle]
pub extern "C" fn choose_move(max_depth: u32, node_budget: u32) -> u32 {
    let buf = unsafe { &*core::ptr::addr_of!(IN_BUF) };
    let mut s = State {
        board: [0; 25],
        cr: buf[25] as i8,
        cc: buf[26] as i8,
        turn: buf[27],
        br: buf[28] as i8,
        bc: buf[29] as i8,
    };
    s.board.copy_from_slice(&buf[..25]);
    // Banned center: both bytes 0 means none; anything else must be a valid
    // center or the input is rejected below.
    if s.br == 0 && s.bc == 0 {
        s.br = -1;
        s.bc = -1;
    } else if !(1..=3).contains(&s.br) || !(1..=3).contains(&s.bc) {
        return NO_MOVE;
    }
    // Reject garbage input rather than risking out-of-bounds panics. Board
    // cells must be 0/1/2 with at most 4 pieces per side (this also bounds
    // legal_moves well under its output buffer).
    if !(1..=3).contains(&s.cr) || !(1..=3).contains(&s.cc) || !(1..=2).contains(&s.turn) {
        return NO_MOVE;
    }
    if s.board.iter().any(|&v| v > 2) || placed(&s, 1) > PIECES || placed(&s, 2) > PIECES {
        return NO_MOVE;
    }
    let seed = unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        SEED
    };
    choose(&s, max_depth.min(8) as i32, node_budget as i64, seed)
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
