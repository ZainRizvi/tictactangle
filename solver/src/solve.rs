//! Exact game-value solver via forward fixpoint labeling for a loopy game.
//!
//! Value is stored 1 byte per state (RAM is ample) with the meaning, always
//! from the *active* (to-move) player's perspective:
//!   0 = UNKNOWN  (also the final label for DRAW positions)
//!   1 = WIN      (active can force a win)
//!   2 = LOSS     (active loses under optimal play)
//!
//! Labeling rule at a state s, over its legal moves:
//!   - s is WIN if some move immediately wins for active, or leads to a child
//!     (the opponent to move) already labeled LOSS.
//!   - s is LOSS if every move either immediately loses (active's slide reveals
//!     only the opponent's line) or leads to a child already labeled WIN, and
//!     there is no immediate-win move and no tie move available (a tie move
//!     guarantees at least a draw, forbidding LOSS).
//!   - otherwise s stays UNKNOWN (draw at fixpoint).
//!
//! We iterate passes over all UNKNOWN states until no label changes. WIN labels
//! flow outward from terminal wins; LOSS labels settle once every escape is a
//! proven WIN for the opponent.

use crate::canon::Canon;
use crate::index::{Indexer, BLOCKS};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use tictactwo_engine::{apply, legal_moves, Move, Outcome, State};

pub const UNKNOWN: u8 = 0;
pub const WIN: u8 = 1;
pub const LOSS: u8 = 2;

pub struct Solver {
    pub ix: Indexer,
    /// One byte per state; indexed by the global perfect index.
    pub values: Vec<AtomicU8>,
    pub total: u64,
}

/// Classify one state from the current value table. Returns the new label
/// (UNKNOWN/WIN/LOSS). Pure read of children; does not mutate.
#[inline]
fn classify(s: &State, ix: &Indexer, values: &[AtomicU8]) -> u8 {
    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    // A to-move position always has at least one legal move (placements early,
    // slides/moves later); n==0 would be a decided/degenerate state we skip.
    debug_assert!(n > 0);

    let mut all_children_win = true; // every move => we lose (child WIN or immediate loss)

    for &m in &moves[..n] {
        let (ns, out) = apply(s, m);
        match out {
            Outcome::Win(1) => return WIN, // active wins immediately
            Outcome::Win(_) => {
                // active's own move revealed opponent's line: this move loses.
                // Does not help toward WIN and counts as "we lose here" for the
                // all-children-win test.
            }
            Outcome::Tie => {
                all_children_win = false; // a draw escape exists
            }
            Outcome::Undecided => {
                // Child is a to-move position for the opponent.
                let cv = child_value(&ns, ix, values);
                match cv {
                    LOSS => return WIN,        // opponent loses => we win
                    WIN => {}                  // opponent wins => bad for us
                    _ => all_children_win = false, // unknown escape
                }
            }
        }
    }

    if all_children_win {
        // Every move loses for us and there is no tie/immediate-win escape.
        LOSS
    } else {
        UNKNOWN
    }
}

/// Look up (or, for junk decided positions, treat as UNKNOWN) the stored value
/// of a child to-move position.
#[inline]
fn child_value(ns: &State, ix: &Indexer, values: &[AtomicU8]) -> u8 {
    // Undecided children are always valid to-move positions with valid counts.
    let c = Canon::from_state(ns).expect("undecided child is a valid to-move state");
    let idx = c.index(ix) as usize;
    values[idx].load(Ordering::Relaxed)
}

impl Solver {
    pub fn new() -> Self {
        let ix = Indexer::new();
        let total = ix.total();
        let mut values = Vec::with_capacity(total as usize);
        values.resize_with(total as usize, || AtomicU8::new(UNKNOWN));
        Solver { ix, values, total }
    }

    /// Enumerate the (block, local) range for a block as global indices.
    fn block_range(&self, block: usize) -> (u64, u64) {
        let base = self.ix.block_base(block);
        let size = self.ix.block_size(block);
        (base, base + size)
    }

    /// Classify one block's UNKNOWN states in parallel; store new labels and
    /// return (new_wins, new_losses) for this block.
    fn pass_block(&self, block: usize) -> (u64, u64) {
        let new_wins = AtomicU64::new(0);
        let new_losses = AtomicU64::new(0);
        let values = &self.values;
        let ix = &self.ix;
        let (lo, hi) = self.block_range(block);
        let base = self.ix.block_base(block);
        (lo..hi).into_par_iter().for_each(|g| {
            if values[g as usize].load(Ordering::Relaxed) != UNKNOWN {
                return;
            }
            let local = g - base;
            let mut active = [0u8; 4];
            let mut waiting = [0u8; 4];
            let (na, nw, center) = ix.deindex(block, local, &mut active, &mut waiting);
            let c = Canon {
                active,
                na,
                waiting,
                nw,
                center,
            };
            let s = c.to_state();
            // Skip positions that already contain a completed line (they can
            // never be genuine to-move states); leave them UNKNOWN.
            if has_any_line(&s) {
                return;
            }
            let label = classify(&s, ix, values);
            if label != UNKNOWN {
                values[g as usize].store(label, Ordering::Relaxed);
                if label == WIN {
                    new_wins.fetch_add(1, Ordering::Relaxed);
                } else {
                    new_losses.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        (
            new_wins.load(Ordering::Relaxed),
            new_losses.load(Ordering::Relaxed),
        )
    }

    /// For each block, the set of blocks its children fall into. A state in
    /// block (a,b) has children in (b,a) [slide/piece-move, counts unchanged,
    /// turn flips] and (b,a+1) [placement, active count +1, turn flips]. A block
    /// only needs rescanning when it, or one of these child-blocks, changed in
    /// the previous pass (a parent can only flip once a child's label appears).
    fn child_blocks(&self) -> Vec<Vec<usize>> {
        let find = |a: u32, b: u32| BLOCKS.iter().position(|&p| p == (a, b));
        (0..BLOCKS.len())
            .map(|i| {
                let (a, b) = BLOCKS[i];
                let mut deps = Vec::new();
                if let Some(j) = find(b, a) {
                    deps.push(j);
                }
                if let Some(j) = find(b, a + 1) {
                    deps.push(j);
                }
                deps
            })
            .collect()
    }

    /// Iterate passes until the fixpoint is reached. `log` prints per-pass stats
    /// including a per-block breakdown. Uses a block-frontier: after the first
    /// full pass, a block is only rescanned when it or one of its child-blocks
    /// changed in the previous pass.
    ///
    /// Crash/session safety: at each pass boundary the whole table plus a small
    /// sidecar (pass number + per-block changed flags) are checkpointed to
    /// `ckpt_path` via atomic rename, and a run resumes from the latest
    /// checkpoint if one exists. `table_path` receives the final table.
    pub fn solve(
        &self,
        table_path: &str,
        ckpt_path: &str,
        log: impl Fn(usize, u64, u64, std::time::Duration, &str),
    ) {
        let deps = self.child_blocks();
        let nb = BLOCKS.len();
        // Resume from a checkpoint if present, else start fresh (all dirty).
        let (mut pass_no, mut changed_prev) = match self.load_checkpoint(ckpt_path, nb) {
            Some((p, cp)) => {
                eprintln!("Resuming from checkpoint at pass {} (table loaded)", p);
                (p, cp)
            }
            None => (0usize, vec![true; nb]),
        };
        loop {
            pass_no += 1;
            let t0 = std::time::Instant::now();
            let mut changed_now = vec![false; nb];
            let mut total_w = 0u64;
            let mut total_l = 0u64;
            let mut detail = String::new();
            for block in 0..nb {
                // Dirty if this block or any of its child-blocks changed last pass.
                let dirty = changed_prev[block] || deps[block].iter().any(|&d| changed_prev[d]);
                if !dirty {
                    continue;
                }
                let tb = std::time::Instant::now();
                let (w, l) = self.pass_block(block);
                let (a, b) = BLOCKS[block];
                // Live per-block line so the huge blocks show progress mid-pass.
                eprintln!(
                    "    [pass {:>3}] block ({},{}): +{:>10}W +{:>10}L  ({:.1?})",
                    pass_no,
                    a,
                    b,
                    w,
                    l,
                    tb.elapsed()
                );
                if w + l > 0 {
                    changed_now[block] = true;
                    detail.push_str(&format!(" ({},{}):+{}W/+{}L", a, b, w, l));
                }
                total_w += w;
                total_l += l;
            }
            let dt = t0.elapsed();
            log(pass_no, total_w, total_l, dt, &detail);
            changed_prev = changed_now;
            let converged = total_w == 0 && total_l == 0;
            // Checkpoint the pass boundary. On convergence we still checkpoint
            // (so a resumed run detects it's done) and write the final table.
            let tc = std::time::Instant::now();
            if let Err(e) = self.save_checkpoint(ckpt_path, pass_no, &changed_prev) {
                eprintln!("WARN: checkpoint write failed: {}", e);
            } else {
                eprintln!("    checkpoint saved after pass {} ({:.1?})", pass_no, tc.elapsed());
            }
            if converged {
                if let Err(e) = self.save(table_path) {
                    eprintln!("WARN: final table write failed: {}", e);
                }
                break;
            }
        }
    }

    /// Count WIN / LOSS / (UNKNOWN=DRAW) across the whole table.
    pub fn tally(&self) -> (u64, u64, u64) {
        let wins = AtomicU64::new(0);
        let losses = AtomicU64::new(0);
        (0..self.total).into_par_iter().for_each(|g| {
            match self.values[g as usize].load(Ordering::Relaxed) {
                WIN => {
                    wins.fetch_add(1, Ordering::Relaxed);
                }
                LOSS => {
                    losses.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        });
        let w = wins.load(Ordering::Relaxed);
        let l = losses.load(Ordering::Relaxed);
        (w, l, self.total - w - l)
    }

    #[inline]
    pub fn value_of(&self, c: &Canon) -> u8 {
        self.values[c.index(&self.ix) as usize].load(Ordering::Relaxed)
    }

    /// The value bytes as a contiguous slice.
    /// SAFETY: AtomicU8 is repr(transparent) over u8, so the Vec<AtomicU8>
    /// backing store is a contiguous [u8] with identical layout.
    #[inline]
    fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.values.as_ptr() as *const u8, self.values.len())
        }
    }

    /// Persist the raw value bytes to disk atomically (write to `<path>.tmp`
    /// then rename). Returns the byte count written.
    pub fn save(&self, path: &str) -> std::io::Result<usize> {
        use std::io::Write;
        let bytes = self.bytes();
        let tmp = format!("{}.tmp", path);
        {
            let f = std::fs::File::create(&tmp)?;
            let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
            w.write_all(bytes)?;
            w.flush()?;
            w.into_inner()?.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(bytes.len())
    }

    /// Write a pass-boundary checkpoint: the full table at `ckpt` (atomic) and
    /// a tiny sidecar `<ckpt>.meta` with the pass number and per-block changed
    /// flags. The table is written first, then the meta, so a meta present on
    /// disk always refers to a fully-written table.
    fn save_checkpoint(&self, ckpt: &str, pass_no: usize, changed: &[bool]) -> std::io::Result<()> {
        use std::io::Write;
        self.save(ckpt)?;
        let meta = format!("{}.meta", ckpt);
        let tmp = format!("{}.tmp", meta);
        {
            let mut f = std::fs::File::create(&tmp)?;
            // Format: "total pass_no\nflags(0/1 per block)\n"
            writeln!(f, "{} {}", self.total, pass_no)?;
            let flags: String = changed.iter().map(|&c| if c { '1' } else { '0' }).collect();
            writeln!(f, "{}", flags)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &meta)?;
        Ok(())
    }

    /// Load a checkpoint's table into `self.values` and return (pass_no,
    /// changed_prev) if a valid, matching checkpoint exists. Mutates the table
    /// in place. Returns None if there is no checkpoint or it doesn't match.
    fn load_checkpoint(&self, ckpt: &str, nb: usize) -> Option<(usize, Vec<bool>)> {
        let meta = format!("{}.meta", ckpt);
        let meta_txt = std::fs::read_to_string(&meta).ok()?;
        let mut lines = meta_txt.lines();
        let header = lines.next()?;
        let mut it = header.split_whitespace();
        let total: u64 = it.next()?.parse().ok()?;
        let pass_no: usize = it.next()?.parse().ok()?;
        if total != self.total {
            eprintln!("checkpoint total {} != current {}; ignoring", total, self.total);
            return None;
        }
        let flags_line = lines.next()?;
        if flags_line.len() != nb {
            return None;
        }
        let changed: Vec<bool> = flags_line.chars().map(|c| c == '1').collect();
        // Load the table bytes into the (already allocated) values in place.
        let bytes = std::fs::read(ckpt).ok()?;
        if bytes.len() as u64 != self.total {
            eprintln!("checkpoint table size mismatch; ignoring");
            return None;
        }
        for (dst, &b) in self.values.iter().zip(bytes.iter()) {
            dst.store(b, Ordering::Relaxed);
        }
        Some((pass_no, changed))
    }

    /// Load a value table from disk. Must match the current index total.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let ix = Indexer::new();
        let total = ix.total();
        let bytes = std::fs::read(path)?;
        if bytes.len() as u64 != total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "table size {} != expected {} (rebuild)",
                    bytes.len(),
                    total
                ),
            ));
        }
        let mut values = Vec::with_capacity(total as usize);
        values.extend(bytes.into_iter().map(AtomicU8::new));
        Ok(Solver { ix, values, total })
    }
}

/// True if either player already has a completed 3-in-a-row inside the grid.
pub fn has_any_line(s: &State) -> bool {
    // Reuse the engine's outcome via a probe: a state with a line is decided.
    // We inline the check to avoid depending on private engine internals.
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
    for p in 1..=2u8 {
        for line in &LINES {
            if line.iter().all(|&(dr, dc)| {
                let r = s.cr + dr;
                let c = s.cc + dc;
                s.board[(r * 5 + c) as usize] == p
            }) {
                return true;
            }
        }
    }
    false
}

/// Given a solved table, return the optimal moves from a position (all moves
/// achieving the best achievable value) along with the position's own value.
pub fn optimal_moves(s: &State, solver: &Solver) -> (u8, Vec<Move>) {
    let mut moves = [0u32; 96];
    let n = legal_moves(s, &mut moves);
    // Rank the outcome of each move from the mover's perspective:
    //   best (2) = immediate win or child LOSS  -> our WIN
    //   mid  (1) = tie, or child UNKNOWN (draw)  -> our DRAW
    //   worst(0) = immediate self-loss, or child WIN -> our LOSS
    let mut best_rank = 0u8;
    let mut ranked: Vec<(u8, Move)> = Vec::with_capacity(n);
    for &m in &moves[..n] {
        let (ns, out) = apply(s, m);
        let rank = match out {
            Outcome::Win(1) => 2,
            Outcome::Win(_) => 0,
            Outcome::Tie => 1,
            Outcome::Undecided => {
                let c = Canon::from_state(&ns).expect("valid child");
                match solver.value_of(&c) {
                    LOSS => 2,
                    WIN => 0,
                    _ => 1,
                }
            }
        };
        if rank > best_rank {
            best_rank = rank;
        }
        ranked.push((rank, m));
    }
    let our_value = match best_rank {
        2 => WIN,
        1 => UNKNOWN, // draw
        _ => LOSS,
    };
    let best: Vec<Move> = ranked
        .into_iter()
        .filter(|&(r, _)| r == best_rank)
        .map(|(_, m)| m)
        .collect();
    (our_value, best)
}
