//! Conversion between the engine's `State` (X/O with an absolute turn) and the
//! solver's normalized canonical form (active-to-move vs. waiting), and the
//! mapping to/from the global perfect index.
//!
//! We solve the **no-ban** official ruleset: any move-repetition ban the engine
//! might track (its `ls`/`bn` anti-loop fields, or any future accumulating ban
//! list) is normalized away — `from_state` ignores those fields and `to_state`
//! clears them, so `apply`/`legal_moves` never arm or enforce a ban. This
//! `no_ban` normalization is applied to every state (including every child)
//! before it is encoded, which is exactly the game with no move banning.

use crate::index::{center_code_rc, center_rc, Indexer, BLOCKS};
use tictactwo_engine::State;

/// A normalized position: occupancy sets for the player to move (active) and
/// the player who just moved (waiting), plus the grid center as 0..8. No ban
/// state — this is the no-ban ruleset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canon {
    /// Ascending occupied squares of the active player.
    pub active: [u8; 4],
    pub na: usize,
    /// Ascending occupied squares of the waiting player.
    pub waiting: [u8; 4],
    pub nw: usize,
    /// Grid center encoded as (cr-1)*3 + (cc-1), range 0..9.
    pub center: u32,
}

/// Encode a grid center (cr, cc), each in 1..=3, to 0..9.
#[inline]
pub fn center_code(cr: i8, cc: i8) -> u32 {
    center_code_rc(cr, cc)
}

/// Decode 0..9 back to (cr, cc).
#[inline]
pub fn center_decode(code: u32) -> (i8, i8) {
    center_rc(code)
}

/// Find the block index for the given (active, waiting) counts, or None if the
/// pair is not a reachable alternation state.
#[inline]
pub fn block_for(na: usize, nw: usize) -> Option<usize> {
    BLOCKS
        .iter()
        .position(|&(a, b)| a as usize == na && b as usize == nw)
}

/// Build an engine `State` with all ban-tracking fields cleared (the no_ban
/// normalization). Centralized so a future engine change to the ban fields is a
/// one-line fix here rather than scattered across the crate.
pub fn ban_free_state(board: [u8; 25], cr: i8, cc: i8, turn: u8) -> State {
    State {
        board,
        cr,
        cc,
        turn,
        ls_r: -1,
        ls_c: -1,
        bn: 0,
        bn_prev: 0,
    }
}

impl Canon {
    /// Build a canonical position from an engine state (the `no_ban`
    /// normalization: any ls/bn ban-tracking fields are ignored). The active
    /// player is `s.turn`; the waiting player is the other. Returns None if the
    /// piece counts violate alternation (not a reachable to-move position).
    pub fn from_state(s: &State) -> Option<Canon> {
        let me = s.turn;
        let opp = 3 - me;
        let mut active = [0u8; 4];
        let mut waiting = [0u8; 4];
        let mut na = 0usize;
        let mut nw = 0usize;
        for sq in 0..25u8 {
            match s.board[sq as usize] {
                v if v == me => {
                    if na >= 4 {
                        return None;
                    }
                    active[na] = sq;
                    na += 1;
                }
                v if v == opp => {
                    if nw >= 4 {
                        return None;
                    }
                    waiting[nw] = sq;
                    nw += 1;
                }
                _ => {}
            }
        }
        block_for(na, nw)?;
        Some(Canon {
            active,
            na,
            waiting,
            nw,
            center: center_code(s.cr, s.cc),
        })
    }

    /// Reconstruct an engine state. We always assign the active player X (turn
    /// 1) and the waiting player O (turn 2); the game value is invariant to the
    /// swap.
    ///
    /// The `no_ban` normalization clears every ban-tracking field: `ls_r/ls_c`
    /// = -1 (no last slide, so `apply` never detects an undo and never arms a
    /// ban) and the ban masks `bn`/`bn_prev` = 0 (no ban enforced). With this
    /// invariant on every state fed to `apply`, no ban is ever armed or
    /// enforced, so we solve the plain official ruleset with no move banning.
    /// If the engine grows more ban fields, clear them here too.
    pub fn to_state(&self) -> State {
        let mut board = [0u8; 25];
        for &sq in &self.active[..self.na] {
            board[sq as usize] = 1;
        }
        for &sq in &self.waiting[..self.nw] {
            board[sq as usize] = 2;
        }
        let (cr, cc) = center_decode(self.center);
        State {
            board,
            cr,
            cc,
            turn: 1,
            ls_r: -1,
            ls_c: -1,
            bn: 0,
            bn_prev: 0,
        }
    }

    #[inline]
    pub fn block(&self) -> usize {
        block_for(self.na, self.nw).expect("valid counts")
    }

    /// Global perfect index of this position (center is the only per-position
    /// grid dimension in the no-ban ruleset).
    pub fn index(&self, ix: &Indexer) -> u64 {
        ix.index(
            self.block(),
            &self.active[..self.na],
            &self.waiting[..self.nw],
            self.center,
        )
    }

    /// Build a canonical position from a global index. Convenience for the
    /// query path and verification.
    pub fn from_index(ix: &Indexer, global: u64) -> Canon {
        // Locate block: largest base <= global.
        let mut block = 0usize;
        for b in 0..BLOCKS.len() {
            if ix.block_base(b) <= global {
                block = b;
            } else {
                break;
            }
        }
        let local = global - ix.block_base(block);
        let mut active = [0u8; 4];
        let mut waiting = [0u8; 4];
        let (na, nw, center) = ix.deindex(block, local, &mut active, &mut waiting);
        Canon {
            active,
            na,
            waiting,
            nw,
            center,
        }
    }
}
