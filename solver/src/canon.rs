//! Conversion between the engine's `State` (X/O with an absolute turn, plus the
//! anti-loop slide history) and the solver's normalized canonical form
//! (active-to-move vs. waiting), and the mapping to/from the global perfect
//! index.

use crate::index::{center_code_rc, center_rc, slide_legal_block, Indexer, SlideState, BLOCKS};
use tictactwo_engine::State;

/// A normalized position: occupancy sets for the player to move (active) and
/// the player who just moved (waiting), the grid center as 0..8, and the
/// anti-loop slide state (last-slide origin + one-turn ban).
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
    /// Anti-loop slide state (ls origin + ban). For blocks where slides are not
    /// yet legal this is always {none, none}.
    pub ss: SlideState,
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

/// Map an engine center field pair (-1 = none, else 1..3) to an optional code.
#[inline]
fn field_center(r: i8, c: i8) -> Option<u32> {
    if r < 0 || c < 0 {
        None
    } else {
        Some(center_code_rc(r, c))
    }
}

impl Canon {
    /// Build a canonical position from an engine state. The active player is
    /// `s.turn`; the waiting player is the other. Returns None if the piece
    /// counts violate alternation (not a reachable to-move position).
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
        let block = block_for(na, nw)?;
        // Slide state only exists where slides are legal; otherwise it must be
        // empty (a placement/early position can carry no ls/ban).
        let ss = if slide_legal_block(na as u32, nw as u32) {
            SlideState {
                ls: field_center(s.ls_r, s.ls_c),
                ban: field_center(s.bn_r, s.bn_c),
            }
        } else {
            SlideState { ls: None, ban: None }
        };
        let _ = block;
        Some(Canon {
            active,
            na,
            waiting,
            nw,
            center: center_code(s.cr, s.cc),
            ss,
        })
    }

    /// Reconstruct an engine state. We always assign the active player X (turn
    /// 1) and the waiting player O (turn 2); this is a canonical representative
    /// — the game value is invariant to the swap. The anti-loop fields are set
    /// from the slide state (-1 = none).
    pub fn to_state(&self) -> State {
        let mut board = [0u8; 25];
        for &sq in &self.active[..self.na] {
            board[sq as usize] = 1;
        }
        for &sq in &self.waiting[..self.nw] {
            board[sq as usize] = 2;
        }
        let (cr, cc) = center_decode(self.center);
        let (ls_r, ls_c) = match self.ss.ls {
            Some(code) => center_rc(code),
            None => (-1, -1),
        };
        let (bn_r, bn_c) = match self.ss.ban {
            Some(code) => center_rc(code),
            None => (-1, -1),
        };
        State {
            board,
            cr,
            cc,
            turn: 1,
            ls_r,
            ls_c,
            bn_r,
            bn_c,
        }
    }

    #[inline]
    pub fn block(&self) -> usize {
        block_for(self.na, self.nw).expect("valid counts")
    }

    /// The combined center+slide-state code for this position's block.
    #[inline]
    fn cb(&self, ix: &Indexer) -> u32 {
        if slide_legal_block(self.na as u32, self.nw as u32) {
            ix.slide_states().combine(self.center, self.ss)
        } else {
            self.center
        }
    }

    /// Global perfect index of this position.
    pub fn index(&self, ix: &Indexer) -> u64 {
        ix.index(
            self.block(),
            &self.active[..self.na],
            &self.waiting[..self.nw],
            self.cb(ix),
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
        let (na, nw, cb) = ix.deindex(block, local, &mut active, &mut waiting);
        let (center, ss) = if slide_legal_block(na as u32, nw as u32) {
            ix.slide_states().split(cb)
        } else {
            (cb, SlideState { ls: None, ban: None })
        };
        Canon {
            active,
            na,
            waiting,
            nw,
            center,
            ss,
        }
    }
}
