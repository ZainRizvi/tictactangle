# Tic Tac Two exact solver

A native Rust crate that computes the game-theoretic value (WIN / LOSS / DRAW
under optimal play) of **every reachable position** of Tic Tac Two, then reports
the value of the initial position and the set of optimal first moves.

It depends read-only on the game engine (`tictactwo-engine`, in `../wasm/engine`)
for the rules, move generation, and move application — the solver never
reimplements the rules, it only enumerates and labels states.

## The game (no-ban official ruleset)

- 5×5 board; a movable 3×3 grid whose center sits at `(cr, cc)`, each in `1..=3`
  (9 grid centers). Only the 9 cells under the grid can win.
- 4 pieces per player; pieces never leave the board.
- Each player's first two turns must be placements. Non-placement moves (slide
  the grid one step, or move one of your own pieces to an empty grid cell)
  unlock only once **both** players have ≥2 pieces down.
- Win = three of your own pieces in a line **inside the current grid**, checked
  after every move. A move lighting lines for **both** players at once is a tie.
  A slide revealing only the **opponent's** line makes the mover lose.

This solver models the game with **no move banning** — the plain official rules.
The shipped engine carries an anti-loop rule (a move-repetition ban tracked in
its `ls`/`bn` state fields), but we normalize it away: `Canon::from_state`
ignores those fields and `Canon::to_state` clears them, so the engine's
`apply`/`legal_moves` never arm or enforce a ban. Because this `no_ban`
normalization is applied to every position (including every child) before it is
encoded, the solved game is exactly the one with no banning, and the game state
is fully described by board + grid center. (`ban_free_state` centralizes the
field-clearing so a future change to the engine's ban fields is a one-line fix.)

## State-space math

We normalize by **turn symmetry**: the value of a position depends only on the
occupancy of the *player to move* ("active") vs. the *player who just moved*
("waiting"), never on the X/O labels. A normalized state is

    (activeMask: 25-bit set, waitingMask: 25-bit set, center: 0..8)

with the two masks disjoint.

### Piece-count blocks (13, not 9)

Turn alternation constrains the counts `(a, b) = (|active|, |waiting|)`, but not
to `|a − b| ≤ 1` as one might first guess. Each player's first two turns are
*forced* placements, but after that placement is optional — a player can stop at
2 pieces and spend later turns sliding/moving while the opponent places all 4.
So the count gap reaches 2. A BFS over the move rules (verified against the
engine) gives exactly **13** reachable count blocks:

    (0,0) (0,1) (1,1) (1,2) (2,2) (2,3) (2,4) (3,2) (3,3) (3,4) (4,2) (4,3) (4,4)

including the asymmetric `(2,4)`, `(3,2)`, `(4,2)`, `(4,3)` the naive assumption
misses. The board-configuration count (before the grid center) is

    Σ over blocks  C(25, a) · C(25 − a, b)  =  119,360,276

Each position then multiplies by the 9-way grid center:

| block | configs      | × 9 centers   |
|-------|--------------|---------------|
| (0,0) | 1            | 9             |
| (0,1) | 25           | 225           |
| (1,1) | 600          | 5,400         |
| (1,2) | 6,900        | 62,100        |
| (2,2) | 75,900       | 683,100       |
| (2,3) | 531,300      | 4,781,700     |
| (2,4) | 2,656,500    | 23,908,500    |
| (3,2) | 531,300      | 4,781,700     |
| (3,3) | 3,542,000    | 31,878,000    |
| (3,4) | 16,824,500   | 151,420,500   |
| (4,2) | 2,656,500    | 23,908,500    |
| (4,3) | 16,824,500   | 151,420,500   |
| (4,4) | 75,710,250   | 681,392,250   |

- **configs: 119,360,276**
- **× 9 grid centers: 1,074,242,484 total states**

At one byte per state the value table is ~1.07 GB.

The naive estimate was **119,999,650** — raw board colorings ignoring the grid
and alternation. The exact config count 119,360,276 lands near it by coincidence
(alternation trims some colorings; the naive figure never counted the grid), and
the real total is ~9× larger once the ×9 grid center is included.

### Indexing (perfect, dense)

Each state maps to a unique integer in `[0, 1,074,242,484)`:

    index = block_base
          + center
          + 9 · ( rank(active in C(25,a))
                  + C(25,a) · rank(waiting in C(25−a,b)) )

`rank(·)` is the [combinatorial number system][cns] rank (active among all
`a`-subsets of 25 squares; waiting among the `b`-subsets of the `25−a` non-active
squares, re-indexed densely). `center` is 0..9. The mapping is a verified
bijection onto the dense range, so the table is a flat `Vec<u8>`.

[cns]: https://en.wikipedia.org/wiki/Combinatorial_number_system

## Algorithm

Tic Tac Two is a **loopy** game (grids slide back and forth, pieces move around),
so it is solved by **forward fixpoint labeling** rather than plain backward
induction. Each state's byte is `UNKNOWN=0`, `WIN=1`, or `LOSS=2`, always from
the active player's perspective; `DRAW` is the fixpoint meaning of a state that
stays `UNKNOWN`.

A pass classifies every still-`UNKNOWN` state from its children's current labels,
using the engine's own `legal_moves` / `apply`:

- **WIN** if some legal move immediately wins for the active player, or leads to
  a child (opponent to move) already labeled `LOSS`.
- **LOSS** if *every* legal move either immediately loses (the active player's
  own slide reveals only the opponent's line) or leads to a child already labeled
  `WIN`, **and** there is no immediate-win move and no tie move available (a tie
  guarantees at least a draw, forbidding `LOSS`).
- otherwise it stays `UNKNOWN` (a draw at the fixpoint).

Passes repeat until no label changes. `WIN` flows outward from terminal wins;
`LOSS` settles once every escape is a proven `WIN` for the opponent. This is the
standard least-fixpoint; the remaining `UNKNOWN` states are exactly the draws.

Passes are parallelized across the index space with rayon, resolved states are
skipped, and a **block frontier** skips whole count-blocks whose children did not
change in the previous pass (a block `(a,b)` reads children only from `(b,a)` and
`(b,a+1)`, so it can only flip when one of those — or itself — changed). Positions
that already contain a completed line are skipped as junk; they are never
referenced as `apply` children of a real state.

The table and a pass counter are checkpointed to `<table>.ckpt(.meta)` at every
pass boundary (atomic write + rename), and a run resumes from the latest
checkpoint, so a multi-hour solve survives an interrupted session.

## Correctness verification

- `solver verify` — structural, no table: block sizes sum to the total; the
  index↔state mapping is a verified bijection (round-trip over sampled and
  endpoint indices in every block); `Canon`↔`State` reproduces board and center.
  A **reachability closure** plays tens of thousands of random games from the
  start (driving the raw engine state, ban fields and all) and asserts every
  to-move position and every `Undecided` child canonicalizes to a valid ban-free
  representative that round-trips through the index — the guarantee that
  `(active, waiting, center)` is a complete state under the no-ban normalization
  and children never land outside the table.
- `solver check table.bin` — game values against a solved table: known hand
  positions mirrored from `test/wasm-engine.test.mjs` and `test/game.test.mjs`
  (mate-in-1 by placement, grid slide, and piece move; the tie-slide and
  self-loss-slide positions; the defender position), plus a randomized
  cross-check against the engine's alpha-beta (`choose_scored`): wherever
  alpha-beta *proves* a forced win (score ≥ 1000) the solver must label that
  position `WIN`, and no solver-`LOSS` position may have an alpha-beta-proven win
  for the side to move.

## Running

    cargo run --release -- count            # exact state-space math
    cargo run --release -- verify           # structural self-tests (fast)
    cargo run --release -- solve table.bin  # full solve, report, persist
    cargo run --release -- check table.bin  # game-value correctness checks
    cargo run --release -- query <board25> <cr> <cc> <turn> [table.bin]

The `query` position format matches `tools/` data: a 25-char board string of
`0`/`1`/`2` (row-major, `0`=empty, `1`=X, `2`=O), the grid center row/col (each
`1..3`), and the side to move (`1`=X, `2`=O). The no-ban ruleset carries no ban
state, so no extra fields are needed. It prints the exact value of the side to
move and all optimal moves.

    # initial position
    cargo run --release -- query 0000000000000000000000000 2 2 1 table.bin

## Results

<!-- RESULTS -->
_Filled in after the full solve completes._
