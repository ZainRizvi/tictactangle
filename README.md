# Tic Tac Two

Tic-tac-toe, except the board won't sit still. A static, client-side web game.

**Play it: https://zainrizvi.github.io/tictactangle/**

## The game

Played on a 5×5 board with a movable 3×3 grid (the "spotlight"). Each player has 4 pieces.

- Your first two turns must be placements on empty lit cells.
- Once both players have placed two pieces, a turn is **one** of:
  - **Place** a remaining piece on an empty lit cell.
  - **Slide the grid** one step in any of the 8 directions (it stays on the board).
  - **Move** one of your pieces — from anywhere on the board — to an empty lit cell.
- First player with three-in-a-row of their own pieces *inside the lit grid* wins. Pieces outside the light don't count.
- If a grid slide lights up three-in-a-row for both players at once, the game is a tie.
- Anti-loop: every slide of yours that the opponent undoes joins your banned list and
  may not be repeated; the list keeps growing while the opponent keeps answering with
  undos, and clears as soon as they make any non-undo move. The undo itself is always
  legal. (Kills rotating shuffles like A→B→A→C→A→B, not just the two-cycle.)

Two modes: local two-player, or versus an AI opponent that runs entirely in your browser.

## The AI

The opponent is a Rust engine compiled to WebAssembly: negamax alpha-beta search with a
small neural network as its leaf evaluator, trained to predict the eventual winner from
the side-to-move's perspective. Everything — search and inference — runs client-side in
the WASM module; a pure-JS alpha-beta engine serves as fallback when WASM is unavailable.

Four difficulty levels:

- **Easy**: the medium engine barely allowed to think (depth 2, tiny node budget) — it
  takes obvious wins and blocks one-ply threats but never plans ahead. In engine-vs-engine
  play, medium beats it 18–1.
- **Medium** (`engine-medium.wasm`, ~37 KB): 61→64→32→1 net (~6k parameters) trained on
  ~146k positions of noisy self-play by the JS engine; searches to depth 6 (~150 ms).
  Its evaluator beats both the JS engine and a handcrafted evaluation under identical
  search.
- **Hard** (`engine-hard.wasm`, ~76 KB): second-generation 61→128→64→1 net (~16k
  parameters) trained on ~110k positions of self-play by the *medium engine* — one more
  turn of the self-improvement crank — and searches to depth 8 (~500 ms). Against equal
  opposition it converts wins several plies sooner than medium.
- **Impossible** (`engine-impossible.wasm`, ~477 KB): an AlphaZero-style player. A
  policy+value network (~93k parameters) trained purely by reinforcement learning —
  120 generations of MCTS self-play, no human or engine-labeled data (see `rl/`) —
  playing through MCTS at inference, with an alpha-beta tactical layer that presses
  and converts winning positions. In evaluation it has never lost a game from either
  side (20/20 draws against hard's strongest settings) while matching hard's ability
  to punish weaker opposition (10 wins, 0 losses vs a shallow engine).

A note on measuring strength: the always-available revert slide is a powerful defensive
resource, so games between strong engines gravitate toward long defensive shuffles.
Head-to-head records between strong engines therefore say little — depth of punishment
(how quickly an engine capitalizes on inaccuracies) is where the difficulty levels
differ in practice.

## Architecture

Clean, domain-based layering — the game model is fully decoupled from the UI, so a
different frontend (canvas, terminal, native), online play, or leaderboards can be added
without touching the core.

```
js/
├── domain/rules.js   Pure game model: state, legal moves, win detection.
│                     No DOM, no timers, no I/O. Usable verbatim in Node, workers, tests.
├── app/
│   ├── ports.js      Contracts between layers (PlayerController port).
│   └── session.js    GameSession: authoritative state, turn orchestration, observer
│                     pattern. Seats are pluggable: null = local human, or any
│                     PlayerController (AI engine, network peer, replay…).
├── ai/
│   ├── engine.js     Pure-JS alpha-beta engine implementing PlayerController.
│   ├── wasm.js       Adapter for the WASM engine (Rust search + value net).
│   └── index.js      Engine selection (WASM preferred, JS fallback) and pacing.
├── ui/view.js        DOM view: subscribes to session snapshots, dispatches intents.
│                     Holds only presentation state. Swappable wholesale.
└── main.js           Composition root — the only place layers are wired together.

wasm/
├── engine/           Rust crate: rules, negamax search, embedded value network,
│   │                 and (rl feature) MCTS + policy/value net for impossible.
│   └── weights/      Generated weights modules, one per alpha-beta variant.
├── engine-*.wasm     Built artifacts served to the browser, one per difficulty
└── build.sh          (see wasm/build.sh <variant>).

rl/                   AlphaZero-style training pipeline for the impossible AI:
                      self-play MCTS + policy/value net + Adam + gating, native
                      Rust, no ML dependencies (see rl/README.md).

tools/                Training pipeline (Node, no dependencies):
├── gen-data.mjs      Noisy self-play generator → labeled positions.
├── train.mjs         MLP training (Adam, softsign value head) → weights.json.
├── export-weights.mjs weights.json → wasm/engine/src/weights.rs.
└── arena.mjs         Engine-vs-engine strength matches.
```

Dependency rule: `domain` imports nothing; `app` imports only `domain`; `ai` and `ui`
import `domain`/`app` contracts; `main.js` wires everything.

## Development

The site itself has no build step — plain ES modules plus the committed
`wasm/engine-*.wasm` artifacts.

```bash
npm run serve   # http://localhost:8080
npm test        # domain, session, engine, and wasm-parity tests (node --test)
```

Rebuilding an AI variant (requires Rust with the `wasm32-unknown-unknown` target):

```bash
# 1. self-play data — pick the teacher engine; repeat/parallelize for volume
node tools/gen-data.mjs 1000 data/part-1.txt 1 "wasm:wasm/engine-medium.wasm:5:120000"
# 2. train — pattern, epochs, hidden sizes, output json
node tools/train.mjs "data/part-*.txt" 18 128 64 wasm/engine/weights-hard.json
# 3. codegen + build the variant
node tools/export-weights.mjs wasm/engine/weights-hard.json wasm/engine/weights/hard.rs
./wasm/build.sh hard                             # → wasm/engine-hard.wasm
# 4. strength check (per-color stats + win lengths)
node tools/arena.mjs 20 wasm:wasm/engine-hard.wasm js:650
```

## Credits

Rules as described by [Berkeley GamesCrafters](https://gamescrafters.berkeley.edu/games.php?game=tictactwo).
