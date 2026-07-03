# RL training for Tic Tac Two — the "impossible" AI

An AlphaZero-style reinforcement-learning pipeline, from scratch, no ML
dependencies: a policy+value network trained purely by self-play with
Monte-Carlo tree search. Native Rust for speed; it reuses the exact game
rules from `../wasm/engine` (the same crate the browser engine is built
from), so there is a single rules implementation across trainer and product.

## How it works (the AlphaGo Zero recipe, scaled to this game)

1. **Self-play**: the current best net plays games against itself. Each move
   runs an MCTS of N simulations guided by the net: the policy head proposes
   moves (priors), the value head evaluates leaves, and PUCT balances
   exploration/exploitation. Dirichlet noise at the root and a visit-count
   temperature for the first 12 plies keep the games diverse.
2. **Learning targets**: every position becomes a training sample —
   the MCTS visit distribution is the policy target (search improves on the
   raw policy, so the net learns to predict *improved* play), and the final
   game outcome is the value target.
3. **Training**: Adam on a replay buffer of recent games; loss =
   MSE(value, outcome) + cross-entropy(policy, visit distribution).
4. **Gating**: after each iteration the freshly trained candidate plays the
   current best; it must score ≥55% (draws = ½) to be promoted. This keeps
   the data generator from regressing.
5. Repeat. Each generation's self-play is stronger, so its training data is
   better, so the next net is stronger.

## Architecture

- Net: 61 input features → 128 ReLU trunk → value head (64 ReLU → softsign)
  and policy head (658 logits: 25 placements + 8 slides + 625 from×to piece
  moves), masked softmax over legal moves only. ~93k parameters.
- MCTS: PUCT (c=1.6), perspective-correct backprop (the engine keeps the
  mover's turn on terminal states, so backprop compares sides instead of
  blindly alternating signs), cycle guard for this game's shuffle loops.
- Draws: official rules allow endless grid shuffling; games are cut off at
  100 plies and labeled with the draw-contempt value (see below).

## Usage

```bash
cargo run --release -- loop  [iters] [games/iter] [sims] [threads]
cargo run --release -- eval  <a.bin> <b.bin> [games] [sims]      # net vs net
cargo run --release -- vsab  <model.bin> [games] [sims] [depth] [budget]  # vs alpha-beta
```

Models land in `rl/models/` (gitignored); `best.bin` is the gating champion.
`../wasm/build.sh impossible` embeds `best.bin` into the browser engine
(a copy is committed as `wasm/engine/src/rlnet.bin`), where the same MCTS
plays it at inference time as the **impossible** difficulty.

## Results

The shipped model is generation 120 (60 initial iterations at 200 sims/move,
then a fresh 120-iteration run at 300 sims with draw contempt). Evaluation at
8000 play-time sims:

| opponent | result |
|---|---|
| hard alpha-beta (depth 8, 600k nodes) | 0 W / **0 L** / 20 D |
| shallow alpha-beta (depth 3) | **10 W** / 0 L / 10 D (won every game as X) |

It has never lost a game from either side. Two lessons learned on the way:

1. **Draw contempt** (below) was needed to stop the value head collapsing to
   "everything is a draw" in this heavily drawish game.
2. Even with contempt, pure MCTS played fortress chess: unbeatable but never
   converting. The shipped player is a **hybrid** — alpha-beta scans every
   move for proven forced wins and clearly winning positions (eval ≥ 0.5)
   and presses them; MCTS over the RL net plays everything else. That
   combination matches the alpha-beta engines' punishment while keeping the
   never-loses defense.

## Draw contempt

Official rules are heavily drawish (the revert slide is a permanent defensive
resource). A first training run with draws labeled 0 collapsed into pure
defense: it never lost, and never won either — 20/20 draws even against a
shallow opponent. Draws are therefore labeled −0.15 for *both* sides (in the
training targets and in both search implementations), which keeps the value
head from flattening to zero and makes the policy fight for wins while still
preferring a draw over any loss.
