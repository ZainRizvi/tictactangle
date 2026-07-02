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

Two modes: local two-player, or versus an AI opponent that runs entirely in your browser.

## The AI

The opponent is a Rust engine compiled to WebAssembly: negamax alpha-beta search with a
small neural network (61→64→32→1, ~6k parameters) as its leaf evaluator. The net was
trained on ~143k positions from ~9,600 noisy self-play games, predicting the eventual
winner from the side-to-move's perspective. Everything — search and inference — runs
client-side in the WASM module (~36 KB); a pure-JS alpha-beta engine serves as fallback
when WASM is unavailable. In head-to-head play the trained evaluator beats both the JS
engine and a handcrafted evaluation under identical search.

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
├── engine/           Rust crate: rules, negamax search, embedded value network.
├── engine.wasm       Built artifact served to the browser (see wasm/build.sh).
└── build.sh          cargo build --target wasm32-unknown-unknown + copy.

tools/                Training pipeline (Node, no dependencies):
├── gen-data.mjs      Noisy self-play generator → labeled positions.
├── train.mjs         MLP training (Adam, softsign value head) → weights.json.
├── export-weights.mjs weights.json → wasm/engine/src/weights.rs.
└── arena.mjs         Engine-vs-engine strength matches.
```

Dependency rule: `domain` imports nothing; `app` imports only `domain`; `ai` and `ui`
import `domain`/`app` contracts; `main.js` wires everything.

## Development

The site itself has no build step — plain ES modules plus a committed `wasm/engine.wasm`.

```bash
npm run serve   # http://localhost:8080
npm test        # domain, session, engine, and wasm-parity tests (node --test)
```

Rebuilding the AI (requires Rust with the `wasm32-unknown-unknown` target):

```bash
node tools/gen-data.mjs 1200 data/part-1.txt 1   # repeat/parallelize for more data
node tools/train.mjs "data/part-*.txt"           # → wasm/engine/weights.json
node tools/export-weights.mjs                    # → wasm/engine/src/weights.rs
./wasm/build.sh                                  # → wasm/engine.wasm
node tools/arena.mjs 20 wasm:wasm/engine.wasm js:650   # strength check
```

## Credits

Rules as described by [Berkeley GamesCrafters](https://gamescrafters.berkeley.edu/games.php?game=tictactwo).
