//! Perfect (minimal, dense) indexing of every reachable Tic Tac Two position
//! under turn symmetry.
//!
//! A position is normalized to the player *to move* ("active") vs. the player
//! who just moved ("waiting") — the X/O identity is irrelevant to the game
//! value, so we fold it away and halve nothing but keep a clean canonical form.
//!
//! State = (active: 25-bit occupancy, waiting: 25-bit occupancy, center: 0..8).
//! The two masks are disjoint. Turn alternation constrains the piece counts to
//!   (|active|, |waiting|) in { (k,k): k=0..4 } U { (k,k+1): k=0..3 }
//! because the active player has either just been handed the move with equal
//! counts, or the waiting player is one placement ahead.
//!
//! Index layout: states are grouped into "blocks" keyed by (a,b) = the piece
//! counts. Within a block the index is
//!   center
//!     + 9 * ( rank_of_active_in_C(25,a)
//!             + C(25,a) * rank_of_waiting_in_C(25-a,b) )
//! plus the block's base offset. rank_of_active is the combinatorial number
//! system rank of the active set among all a-subsets of 25 squares;
//! rank_of_waiting is the CNS rank of the waiting set among all b-subsets of
//! the 25-a squares *not* used by active, re-indexed densely to 0..(25-a).
//!
//! Reachable (|active|, |waiting|) count pairs: a player's first two turns are
//! forced placements, but afterward placement is optional, so a player may stop
//! at 2 pieces while the other reaches 4 — the count gap can be up to 2. The 13
//! reachable pairs (verified by game BFS and by the engine at solve time) are
//! below; the naive |a-b|<=1 assumption misses the asymmetric ones like (2,4).

pub const NBLOCKS: usize = 13;

/// The 13 reachable count blocks. `.0` = |active|, `.1` = |waiting|.
pub const BLOCKS: [(u32, u32); NBLOCKS] = [
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
    (4, 4),
];

pub const CENTERS: u64 = 9;
pub const SQUARES: usize = 25;

/// Decode center code 0..9 to (cr, cc), each 1..3.
#[inline]
pub fn center_rc(code: u32) -> (i8, i8) {
    ((code / 3) as i8 + 1, (code % 3) as i8 + 1)
}

/// Encode (cr, cc) to center code 0..9.
#[inline]
pub fn center_code_rc(cr: i8, cc: i8) -> u32 {
    ((cr - 1) * 3 + (cc - 1)) as u32
}


/// Binomial coefficients C(n,k) for n,k <= 25. C[n][k].
pub struct Binom {
    table: [[u64; 26]; 26],
}

impl Binom {
    pub fn new() -> Self {
        let mut table = [[0u64; 26]; 26];
        for n in 0..=25usize {
            table[n][0] = 1;
            for k in 1..=n {
                table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            }
        }
        Binom { table }
    }

    #[inline]
    pub fn c(&self, n: usize, k: usize) -> u64 {
        if k > n {
            0
        } else {
            self.table[n][k]
        }
    }
}

/// Precomputed offsets and sizes so global index <-> (block, local) is O(1).
///
/// No-ban ruleset: the only per-position grid dimension is the 9-way center, so
/// every block's cb_size is `CENTERS`.
pub struct Indexer {
    binom: Binom,
    /// base[i] = first global index of block i.
    base: [u64; NBLOCKS],
    /// For block i: number of active-subsets = C(25, a).
    active_choices: [u64; NBLOCKS],
    /// For block i: the center dimension size (always 9).
    cb_size: [u64; NBLOCKS],
    /// Total number of states across all blocks.
    total: u64,
}

impl Indexer {
    pub fn new() -> Self {
        let binom = Binom::new();
        let mut base = [0u64; NBLOCKS];
        let mut active_choices = [0u64; NBLOCKS];
        let mut cb_size = [0u64; NBLOCKS];
        let mut acc = 0u64;
        for (i, &(a, b)) in BLOCKS.iter().enumerate() {
            base[i] = acc;
            let ca = binom.c(SQUARES, a as usize);
            let cb = binom.c(SQUARES - a as usize, b as usize);
            active_choices[i] = ca;
            let cbs = CENTERS; // 9-way grid center; no ban dimension
            cb_size[i] = cbs;
            let block_states = ca * cb * cbs;
            acc += block_states;
        }
        Indexer {
            binom,
            base,
            active_choices,
            cb_size,
            total: acc,
        }
    }

    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    #[inline]
    pub fn cb_size(&self, block: usize) -> u64 {
        self.cb_size[block]
    }

    #[inline]
    pub fn block_base(&self, block: usize) -> u64 {
        self.base[block]
    }

    /// Number of states in a block.
    pub fn block_size(&self, block: usize) -> u64 {
        let (a, b) = BLOCKS[block];
        let ca = self.binom.c(SQUARES, a as usize);
        let cb = self.binom.c(SQUARES - a as usize, b as usize);
        ca * cb * self.cb_size[block]
    }

    /// Combinatorial-number-system rank of a k-subset given as sorted squares.
    /// `squares` must be ascending. Rank in [0, C(domain, k)).
    #[inline]
    fn cns_rank(&self, squares: &[u8], _domain: usize) -> u64 {
        // rank = sum over the i-th smallest element x_i of C(x_i, i+1)
        let mut rank = 0u64;
        for (i, &x) in squares.iter().enumerate() {
            rank += self.binom.c(x as usize, i + 1);
        }
        rank
    }

    /// Unrank: produce the ascending k-subset of `domain` squares with the
    /// given CNS rank. Writes into `out[..k]`.
    #[inline]
    fn cns_unrank(&self, mut rank: u64, k: usize, domain: usize, out: &mut [u8]) {
        // Standard greedy CNS unranking, largest element first.
        let mut x = domain; // exclusive upper bound
        for pos in (0..k).rev() {
            // find largest v < x with C(v, pos+1) <= rank
            let mut v = x;
            while v > 0 {
                v -= 1;
                let c = self.binom.c(v, pos + 1);
                if c <= rank {
                    rank -= c;
                    break;
                }
            }
            out[pos] = v as u8;
            x = v;
        }
    }

    /// Compute the global index for a normalized position.
    /// `active` and `waiting` are the sorted-ascending occupied squares of the
    /// active and waiting players; `cb` is the combined center+ban code in
    /// 0..cb_size(block) (0..9 for no-slide blocks, 0..49 for slide-legal ones).
    pub fn index(&self, block: usize, active: &[u8], waiting: &[u8], cb: u32) -> u64 {
        let (a, _b) = BLOCKS[block];
        debug_assert_eq!(active.len(), a as usize);
        debug_assert!((cb as u64) < self.cb_size[block]);

        let active_rank = self.cns_rank(active, SQUARES);

        // Re-index the waiting squares into the complement domain (0..25-a):
        // for each waiting square, subtract the number of active squares below
        // it, giving its position among the non-active squares.
        let mut wre = [0u8; 4];
        for (i, &w) in waiting.iter().enumerate() {
            let below = active.iter().filter(|&&x| x < w).count() as u8;
            wre[i] = w - below;
        }
        let waiting_rank = self.cns_rank(&wre[..waiting.len()], SQUARES - a as usize);

        let ca = self.active_choices[block];
        let combo = active_rank + ca * waiting_rank;
        self.base[block] + cb as u64 + self.cb_size[block] * combo
    }

    /// Inverse of `index` restricted to a known block: recover
    /// (active squares, waiting squares, cb). Returns counts + combined cb code.
    pub fn deindex(
        &self,
        block: usize,
        local: u64,
        active_out: &mut [u8; 4],
        waiting_out: &mut [u8; 4],
    ) -> (usize, usize, u32) {
        let (a, b) = BLOCKS[block];
        let cbs = self.cb_size[block];
        let cb = (local % cbs) as u32;
        let combo = local / cbs;
        let ca = self.active_choices[block];
        let active_rank = combo % ca;
        let waiting_rank = combo / ca;

        self.cns_unrank(active_rank, a as usize, SQUARES, active_out);
        let mut wre = [0u8; 4];
        self.cns_unrank(waiting_rank, b as usize, SQUARES - a as usize, &mut wre);

        // Map re-indexed waiting squares back to absolute squares by walking
        // the non-active squares in order.
        let active = &active_out[..a as usize];
        for i in 0..b as usize {
            let target = wre[i] as usize; // position among non-active squares
            // find the target-th square not in active
            let mut count = 0usize;
            let mut sq = 0usize;
            loop {
                if !active.contains(&(sq as u8)) {
                    if count == target {
                        break;
                    }
                    count += 1;
                }
                sq += 1;
            }
            waiting_out[i] = sq as u8;
        }
        (a as usize, b as usize, cb)
    }
}
