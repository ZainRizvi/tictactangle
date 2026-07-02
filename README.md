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
│   ├── engine.js     Alpha-beta search engine implementing PlayerController.
│   └── index.js      Adapter selection (WASM model preferred, JS fallback).
├── ui/view.js        DOM view: subscribes to session snapshots, dispatches intents.
│                     Holds only presentation state. Swappable wholesale.
└── main.js           Composition root — the only place layers are wired together.
```

Dependency rule: `domain` imports nothing; `app` imports only `domain`; `ai` and `ui`
import `domain`/`app` contracts; `main.js` wires everything.

## Development

No build step — plain ES modules.

```bash
npm run serve   # http://localhost:8080
npm test        # domain + session tests (node --test)
```

## Credits

Rules as described by [Berkeley GamesCrafters](https://gamescrafters.berkeley.edu/games.php?game=tictactwo).
