//! Correctness self-tests.
//!
//! `verify` (no table needed): the index bijection, block sizes, and the
//! canonical-form round trip. These validate the state space plumbing before
//! committing to an ~870M-state solve.
//!
//! `check <table.bin>` (in check.rs): known hand positions and an alpha-beta
//! cross-check, which require a solved table.

use crate::canon::{center_code, Canon};
use crate::index::{Indexer, BLOCKS, SQUARES};
use tictactwo_engine::State;

/// The empty starting position (center (2,2), X to move, no ban state).
fn initial_state() -> State {
    State {
        board: [0; 25],
        cr: 2,
        cc: 2,
        turn: 1,
        ls_r: -1,
        ls_c: -1,
        bn: 0,
        bn_prev: 0,
    }
}

/// A tiny deterministic RNG (xorshift64) so the sampling is reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

pub fn run() {
    println!("=== solver self-tests ===\n");
    let ix = Indexer::new();

    // 1. Block sizes and total.
    let mut acc = 0u64;
    for (i, &(a, b)) in BLOCKS.iter().enumerate() {
        let sz = ix.block_size(i);
        assert_eq!(ix.block_base(i), acc, "block {} base", i);
        acc += sz;
        println!("  block {:?}: {:>12} states", (a, b), sz);
    }
    assert_eq!(acc, ix.total(), "sum of block sizes == total");
    println!("  total: {} states\n", ix.total());

    // 2. Index bijection: for each block, index(deindex(g)) == g for a sample
    //    of local indices spanning the range, plus the endpoints.
    println!("Index round-trip (per-block sampling)...");
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for block in 0..BLOCKS.len() {
        let base = ix.block_base(block);
        let size = ix.block_size(block);
        let samples = 20_000u64.min(size);
        for s in 0..samples {
            // spread: endpoints, then random.
            let local = match s {
                0 => 0,
                1 => size - 1,
                _ => rng.below(size),
            };
            let g = base + local;
            let c = Canon::from_index(&ix, g);
            let (na, nw) = (c.na, c.nw);
            // masks disjoint and ascending
            for w in 1..na {
                assert!(c.active[w] > c.active[w - 1], "active ascending");
            }
            for w in 1..nw {
                assert!(c.waiting[w] > c.waiting[w - 1], "waiting ascending");
            }
            for &pa in &c.active[..na] {
                assert!(!c.waiting[..nw].contains(&pa), "masks disjoint");
                assert!(pa < 25, "square in range");
            }
            let g2 = c.index(&ix);
            assert_eq!(g2, g, "round trip block {} local {}", block, local);
        }
    }
    println!("  OK\n");

    // 3. Canonical from_state / to_state consistency for a few crafted states.
    println!("Canon <-> State consistency...");
    let ix = Indexer::new();
    let mut count = 0;
    // Build random legal-ish positions by placing pieces on distinct squares.
    let mut rng = Rng(0xdead_beef_0000_0001);
    for _ in 0..50_000 {
        let (a, b) = BLOCKS[(rng.below(BLOCKS.len() as u64)) as usize];
        let mut board = [0u8; 25];
        let mut placed = 0u32;
        let target = a + b;
        // choose target distinct squares
        while placed < target {
            let sq = rng.below(25) as usize;
            if board[sq] == 0 {
                board[sq] = if placed < a { 1 } else { 2 };
                placed += 1;
            }
        }
        let center = center_code(
            1 + rng.below(3) as i8,
            1 + rng.below(3) as i8,
        );
        let (cr, cc) = crate::canon::center_decode(center);
        // No-ban ruleset: the canonical state carries no ls/ban; any ban fields
        // on the input are normalized away by from_state.
        let s = tictactwo_engine::State {
            board,
            cr,
            cc,
            turn: 1,
            ls_r: -1,
            ls_c: -1,
            bn: 0,
            bn_prev: 0,
        };
        let c = Canon::from_state(&s).expect("valid counts");
        let g = c.index(&ix);
        let c2 = Canon::from_index(&ix, g);
        assert_eq!(c, c2, "from_index(index(c)) == c");
        // to_state must reproduce the board and center.
        let s2 = c2.to_state();
        assert_eq!(s2.board, board, "board reproduced");
        assert_eq!((s2.cr, s2.cc), (cr, cc), "center reproduced");
        count += 1;
    }
    println!("  OK ({} random positions)\n", count);

    // 3b. Reachability closure: from the initial position, play many random
    //     games; EVERY to-move position and EVERY Undecided child of EVERY
    //     legal move must canonicalize to a valid block. This is the invariant
    //     the solver relies on (children never land outside the table) and it
    //     directly guards against an incomplete BLOCKS set.
    println!("Reachability closure (random playouts, all children in range)...");
    let mut rng = Rng(0x0bad_c0de_face_1234);
    let mut positions = 0u64;
    let mut children = 0u64;
    // Drive the game through ban-free canonical representatives (exactly what
    // the solver explores): each position is the no_ban-normalized `to_state`
    // of its Canon. Every Undecided child must canonicalize and survive an
    // index round-trip reproducing board + center.
    // Driving the *raw* engine state (real turn + real ban fields from the
    // reworked anti-loop rule) and canonicalizing each position is the more
    // thorough test: it exercises genuine reachable states, including
    // ban-carrying ones, and confirms they all canonicalize to a valid ban-free
    // representative that round-trips through the index.
    for _ in 0..30_000 {
        let mut s = initial_state();
        for _ply in 0..80 {
            Canon::from_state(&s).unwrap_or_else(|| {
                panic!(
                    "unreachable to-move block for counts {:?}",
                    (
                        s.board.iter().filter(|&&v| v == s.turn).count(),
                        s.board.iter().filter(|&&v| v == 3 - s.turn).count()
                    )
                )
            });
            positions += 1;
            let mut moves = [0u32; 96];
            let n = tictactwo_engine::legal_moves(&s, &mut moves);
            if n == 0 {
                break;
            }
            for &m in &moves[..n] {
                let (ns, out) = tictactwo_engine::apply(&s, m);
                if out == tictactwo_engine::Outcome::Undecided {
                    let c = Canon::from_state(&ns).unwrap_or_else(|| {
                        panic!(
                            "undecided child out of range: counts {:?}",
                            (
                                ns.board.iter().filter(|&&v| v == ns.turn).count(),
                                ns.board.iter().filter(|&&v| v == 3 - ns.turn).count()
                            )
                        )
                    });
                    let g = c.index(&ix);
                    let back = Canon::from_index(&ix, g).to_state();
                    let cns = c.to_state();
                    assert_eq!(back.board, cns.board, "child board round-trip");
                    assert_eq!((back.cr, back.cc), (cns.cr, cns.cc), "child center");
                    children += 1;
                }
            }
            // advance by a random move, then re-normalize to the ban-free rep
            let m = moves[rng.below(n as u64) as usize];
            let (ns, out) = tictactwo_engine::apply(&s, m);
            if out != tictactwo_engine::Outcome::Undecided {
                break;
            }
            s = ns;
        }
    }
    println!(
        "  OK ({} positions, {} children all in range)\n",
        positions, children
    );

    // 4. Sanity: the initial position lands in block (0,0) at a well-defined
    //    global index within range.
    let init = initial_state();
    let ic = Canon::from_state(&init).unwrap();
    let g = ic.index(&ix);
    assert!(g < ix.total(), "initial index in range");
    // block (0,0) is first; center code for (2,2) is 4.
    assert_eq!(g, 4, "initial position index == center code 4");
    println!("Initial position index = {} (center code 4) OK", g);

    let _ = SQUARES;
    println!("\n=== all structural self-tests passed ===");
}
