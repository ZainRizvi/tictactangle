//! Play-time PUCT MCTS over the RL policy/value net — the "impossible"
//! difficulty. Mirrors the search used during training (rl/src/mcts.rs):
//! same C_PUCT, perspective-correct backprop, cycle guard. No allocation:
//! nodes live in a static pool; children are contiguous (all siblings are
//! created together on expansion).

use crate::rlnet;
use crate::{apply, legal_moves, Move, Outcome, State, NO_MOVE};

const C_PUCT: f32 = 1.6;
const MAX_DESCENT: usize = 160;
// Sized for the UI's 8,000-sim searches at worst-case branching (~45
// children/expansion). Zeroed BSS: costs runtime memory (~25 MB), not
// binary size.
const POOL: usize = 400_000;
/// Draw contempt, matching the training pipeline (rl/src/mcts.rs DRAW_V).
const DRAW_V: f32 = -0.15;
/// Alpha-beta eval (softsign, [-1,1]) above which the tactical layer's
/// pressing move is preferred over MCTS's quiet one.
const PRESS_THRESHOLD: f32 = 0.5;

#[derive(Clone, Copy)]
struct Node {
    state: State,
    mv: Move,
    parent: i32,
    first_child: u32,
    n_children: u16,
    visits: u32,
    value_sum: f32,
    prior: f32,
    terminal: f32, // meaningful when is_terminal
    is_terminal: bool,
    expanded: bool,
}

// All-zero placeholder so the static pool lands in BSS (zero bytes aren't
// stored in the wasm binary). Every node is fully written before first use.
const EMPTY_NODE: Node = Node {
    state: State { board: [0; 25], cr: 0, cc: 0, turn: 0 },
    mv: 0,
    parent: 0,
    first_child: 0,
    n_children: 0,
    visits: 0,
    value_sum: 0.0,
    prior: 0.0,
    terminal: 0.0,
    is_terminal: false,
    expanded: false,
};

static mut NODES: [Node; POOL] = [EMPTY_NODE; POOL];

struct Pool {
    len: usize,
}

#[inline]
fn mean(n: &Node) -> f32 {
    if n.visits == 0 {
        0.0
    } else {
        n.value_sum / n.visits as f32
    }
}

/// Runs `sims` simulations from `root` and returns the most-visited move.
/// `seed` only breaks exact ties among equally visited moves.
///
/// Hybrid tactical layer: alpha-beta first proves or refutes a forced win;
/// MCTS's positional judgment (which never loses) plays everything else.
#[allow(static_mut_refs)]
pub fn choose_mcts(root: &State, sims: u32, seed: u64) -> Move {
    if !rlnet::ensure_ready() {
        return NO_MOVE;
    }
    let (tactic, score) = crate::choose_scored(root, 8, 250_000, seed);
    // Proven forced win, or a clearly winning advantage worth pressing home:
    // use the tactical move. Everything else: MCTS positional judgment.
    if score >= crate::WIN || score >= PRESS_THRESHOLD {
        return tactic;
    }
    let nodes = unsafe { &mut NODES };
    nodes[0] = EMPTY_NODE;
    nodes[0].state = *root;
    nodes[0].prior = 1.0;
    nodes[0].parent = -1; // backprop terminates here (EMPTY_NODE is all-zero)
    let mut pool = Pool { len: 1 };

    let v0 = expand(nodes, &mut pool, 0);
    backprop(nodes, 0, v0);

    let mut done = 1;
    while done < sims {
        // ---- selection ----
        let mut id: usize = 0;
        let mut depth = 0usize;
        loop {
            let node = &nodes[id];
            if node.is_terminal || !node.expanded || node.n_children == 0 || depth >= MAX_DESCENT {
                break;
            }
            let parent_turn = node.state.turn;
            let sqrt_n = rlnet::sqrtf(node.visits.max(1) as f32);
            let first = node.first_child as usize;
            let count = node.n_children as usize;
            let mut best = f32::NEG_INFINITY;
            let mut best_child = first;
            for c in first..first + count {
                let ch = &nodes[c];
                let q = if ch.visits == 0 {
                    0.0
                } else if ch.state.turn == parent_turn {
                    mean(ch)
                } else {
                    -mean(ch)
                };
                let u = C_PUCT * ch.prior * sqrt_n / (1.0 + ch.visits as f32);
                if q + u > best {
                    best = q + u;
                    best_child = c;
                }
            }
            id = best_child;
            depth += 1;
        }

        // ---- evaluate / expand ----
        let value = if nodes[id].is_terminal {
            nodes[id].terminal
        } else if depth >= MAX_DESCENT || pool.len + 96 > POOL {
            DRAW_V // cycle guard or pool exhausted: dead heat
        } else {
            expand(nodes, &mut pool, id)
        };
        backprop(nodes, id, value);
        done += 1;
        if pool.len + 96 > POOL {
            break; // out of memory for further expansion — use what we have
        }
    }

    // ---- pick the most-visited root move (seed breaks ties) ----
    let root_node = &nodes[0];
    let first = root_node.first_child as usize;
    let count = root_node.n_children as usize;
    if count == 0 {
        return NO_MOVE;
    }
    let mut best_visits = 0u32;
    let mut best_mv = NO_MOVE;
    let mut ties = 0u64;
    let mut rng = seed | 1;
    for c in first..first + count {
        let ch = &nodes[c];
        if ch.visits > best_visits {
            best_visits = ch.visits;
            best_mv = ch.mv;
            ties = 1;
        } else if ch.visits == best_visits && best_visits > 0 {
            ties += 1;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng % ties == 0 {
                best_mv = ch.mv;
            }
        }
    }
    best_mv
}

fn expand(nodes: &mut [Node; POOL], pool: &mut Pool, id: usize) -> f32 {
    let state = nodes[id].state;
    let mut moves = [0u32; 96];
    let n = legal_moves(&state, &mut moves);

    let mut actions = [0usize; 96];
    for k in 0..n {
        actions[k] = action_index(moves[k]);
    }
    let mut priors = [0.0f32; 96];
    let value = rlnet::eval(&state, &actions[..n], &mut priors);

    let first = pool.len;
    let parent_turn = state.turn;
    for k in 0..n {
        let (ns, out) = apply(&state, moves[k]);
        let (is_terminal, terminal) = match out {
            Outcome::Undecided => (false, 0.0),
            Outcome::Tie => (true, DRAW_V),
            Outcome::Win(p) => (true, if p == parent_turn { 1.0 } else { -1.0 }),
        };
        nodes[pool.len] = Node {
            state: ns,
            mv: moves[k],
            parent: id as i32,
            first_child: 0,
            n_children: 0,
            visits: 0,
            value_sum: 0.0,
            prior: priors[k],
            terminal,
            is_terminal,
            expanded: false,
        };
        pool.len += 1;
    }
    nodes[id].first_child = first as u32;
    nodes[id].n_children = n as u16;
    nodes[id].expanded = true;
    value
}

fn backprop(nodes: &mut [Node; POOL], leaf: usize, value: f32) {
    let leaf_turn = nodes[leaf].state.turn;
    let mut id = leaf as i32;
    while id >= 0 {
        let node = &mut nodes[id as usize];
        node.value_sum += if node.state.turn == leaf_turn { value } else { -value };
        node.visits += 1;
        id = node.parent;
    }
}

/// Same action indexing as rl/src/game.rs (policy head layout).
fn action_index(m: Move) -> usize {
    const DIR_LIST: [(i8, i8); 8] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ];
    let kind = m >> 16;
    let a = ((m >> 8) & 0xFF) as usize;
    let b = (m & 0xFF) as usize;
    match kind {
        0 => a,
        1 => {
            let dr = a as i8 - 1;
            let dc = b as i8 - 1;
            let mut d = 0;
            for (i, &(r, c)) in DIR_LIST.iter().enumerate() {
                if r == dr && c == dc {
                    d = i;
                    break;
                }
            }
            25 + d
        }
        _ => 33 + a * 25 + b,
    }
}
