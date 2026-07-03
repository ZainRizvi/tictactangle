//! PUCT Monte-Carlo tree search guided by the policy/value net.
//!
//! Sign convention: `value_sum` at a node accumulates simulation results from
//! the perspective of the side to move AT that node (`state.turn`). Terminal
//! positions keep the mover's turn (the engine doesn't flip on game end), so
//! perspective comparisons — not blind sign-flipping per level — are used
//! when propagating and selecting.

use crate::game::{encode, legal_with_indices, FEATS};
use crate::net::Net;
use crate::rng::Rng;
use tictactwo_engine::{apply, Move, Outcome, State};

const C_PUCT: f32 = 1.6;
const MAX_DESCENT: usize = 160;
/// Search-side draw contempt; must match selfplay::DRAW_Z so search values
/// and training labels agree.
const DRAW_V: f32 = -0.15;

pub struct Node {
    pub state: State,
    pub mv: Move, // move that led here (from parent)
    pub parent: i32,
    pub children: Vec<u32>,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    /// Some(v) = terminal, v from the PARENT mover's perspective.
    pub terminal: Option<f32>,
    pub expanded: bool,
}

pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    fn mean(&self, id: u32) -> f32 {
        let n = &self.nodes[id as usize];
        if n.visits == 0 {
            0.0
        } else {
            n.value_sum / n.visits as f32
        }
    }
}

pub struct SearchResult {
    /// (move, visit count) per legal root move.
    pub visits: Vec<(Move, u32)>,
}

/// Runs `sims` simulations from `root_state`. `noise` adds Dirichlet noise to
/// the root priors (self-play exploration).
pub fn search(
    net: &Net,
    root_state: &State,
    sims: u32,
    noise: Option<(f32, f32)>, // (epsilon, alpha)
    rng: &mut Rng,
) -> SearchResult {
    let mut tree = Tree { nodes: Vec::with_capacity(sims as usize * 8) };
    tree.nodes.push(Node {
        state: *root_state,
        mv: 0,
        parent: -1,
        children: Vec::new(),
        prior: 1.0,
        visits: 0,
        value_sum: 0.0,
        terminal: None,
        expanded: false,
    });

    // Expand root immediately so noise can be applied.
    let v0 = expand(&mut tree, 0, net);
    backprop(&mut tree, 0, v0);

    if let Some((eps, alpha)) = noise {
        let k = tree.nodes[0].children.len();
        if k > 1 {
            let dir = rng.dirichlet(alpha, k);
            let child_ids: Vec<u32> = tree.nodes[0].children.clone();
            for (i, &c) in child_ids.iter().enumerate() {
                let p = &mut tree.nodes[c as usize].prior;
                *p = (1.0 - eps) * *p + eps * dir[i];
            }
        }
    }

    for _ in 1..sims {
        // ---- selection ----
        let mut id: u32 = 0;
        let mut depth = 0;
        loop {
            let node = &tree.nodes[id as usize];
            if node.terminal.is_some() || !node.expanded || node.children.is_empty() {
                break;
            }
            if depth >= MAX_DESCENT {
                break; // cycle guard: treat as a dead heat below
            }
            let parent_turn = node.state.turn;
            let sqrt_n = ((node.visits.max(1)) as f32).sqrt();
            let mut best = f32::NEG_INFINITY;
            let mut best_child = node.children[0];
            for &c in &node.children {
                let ch = &tree.nodes[c as usize];
                let q = if ch.visits == 0 {
                    0.0
                } else if ch.state.turn == parent_turn {
                    tree.mean(c)
                } else {
                    -tree.mean(c)
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

        // ---- evaluation / expansion ----
        let value = {
            let node = &tree.nodes[id as usize];
            if let Some(tv) = node.terminal {
                // tv is from the parent mover's perspective; the node's turn
                // equals the mover's, so it's also from node.turn perspective.
                tv
            } else if depth >= MAX_DESCENT {
                DRAW_V
            } else {
                expand(&mut tree, id, net)
            }
        };
        backprop(&mut tree, id, value);
    }

    let visits = tree.nodes[0]
        .children
        .iter()
        .map(|&c| (tree.nodes[c as usize].mv, tree.nodes[c as usize].visits))
        .collect();
    SearchResult { visits }
}

/// Expands `id`: creates children with priors, returns the net's value from
/// the perspective of the side to move at `id`.
fn expand(tree: &mut Tree, id: u32, net: &Net) -> f32 {
    let state = tree.nodes[id as usize].state;
    let (moves, idxs, n) = legal_with_indices(&state);
    let mut feat = [0.0f32; FEATS];
    encode(&state, &mut feat);
    let eval = net.eval(&feat, &idxs[..n]);

    let parent_turn = state.turn;
    for k in 0..n {
        let (ns, out) = apply(&state, moves[k]);
        let ns = crate::game::no_ban(ns);
        let terminal = match out {
            Outcome::Undecided => None,
            Outcome::Tie => Some(DRAW_V),
            Outcome::Win(p) => Some(if p == parent_turn { 1.0 } else { -1.0 }),
        };
        let child = Node {
            state: ns,
            mv: moves[k],
            parent: id as i32,
            children: Vec::new(),
            prior: eval.priors[k],
            visits: 0,
            value_sum: 0.0,
            terminal,
            expanded: false,
        };
        let cid = tree.nodes.len() as u32;
        tree.nodes.push(child);
        tree.nodes[id as usize].children.push(cid);
    }
    tree.nodes[id as usize].expanded = true;
    eval.value
}

/// Adds `value` (from the perspective of the side to move at `leaf`) to every
/// node on the path, converting perspective by comparing turns.
fn backprop(tree: &mut Tree, leaf: u32, value: f32) {
    let leaf_turn = tree.nodes[leaf as usize].state.turn;
    let mut id = leaf as i32;
    while id >= 0 {
        let node = &mut tree.nodes[id as usize];
        let v = if node.state.turn == leaf_turn { value } else { -value };
        node.value_sum += v;
        node.visits += 1;
        id = node.parent;
    }
}
