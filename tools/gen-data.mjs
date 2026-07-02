// Self-play data generator for the value network.
// Plays the JS engine against itself with exploration noise and records every
// position with the eventual outcome from the side-to-move's perspective.
//
// Usage: node tools/gen-data.mjs <games> <outfile> [seed]
// Output: one line per position — "<board25chars> <cr> <cc> <turn> <z>"

import { writeFileSync } from 'node:fs';
import { newGame, applyMove, legalMoves, X } from '../js/domain/rules.js';
import { createJsEngine } from '../js/ai/engine.js';

const games = Number(process.argv[2] ?? 500);
const outfile = process.argv[3] ?? 'data/selfplay.txt';
let rngState = (Number(process.argv[4] ?? 1) * 2654435761) >>> 0 || 1;

const rand = () => {
  // xorshift32
  rngState ^= rngState << 13; rngState >>>= 0;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5; rngState >>>= 0;
  return rngState / 0x100000000;
};

const engine = createJsEngine({ timeBudgetMs: 12 });
const MAX_PLY = 80;
const EPSILON = 0.1;

const lines = [];
let wins = 0, ties = 0, cutoffs = 0;

for (let g = 0; g < games; g++) {
  let s = newGame();
  const positions = []; // {board, cr, cc, turn}
  const openingPlies = 2 + Math.floor(rand() * 4);

  while (!s.result && s.ply < MAX_PLY) {
    positions.push({ board: s.board.join(''), cr: s.center.r, cc: s.center.c, turn: s.turn });
    let move;
    if (s.ply < openingPlies || rand() < EPSILON) {
      const moves = legalMoves(s);
      move = moves[Math.floor(rand() * moves.length)];
    } else {
      move = await engine.chooseMove(s);
    }
    s = applyMove(s, move);
  }

  let winner = 0;
  if (s.result?.type === 'win') { winner = s.result.winner; wins++; }
  else if (s.result?.type === 'tie') ties++;
  else cutoffs++;

  for (const p of positions) {
    const z = winner === 0 ? 0 : winner === p.turn ? 1 : -1;
    lines.push(`${p.board} ${p.cr} ${p.cc} ${p.turn} ${z}`);
  }
  if ((g + 1) % 50 === 0) {
    console.log(`${outfile}: ${g + 1}/${games} games, ${lines.length} positions (w:${wins} t:${ties} c:${cutoffs})`);
  }
}

writeFileSync(outfile, lines.join('\n') + '\n');
console.log(`done: ${outfile} — ${lines.length} positions from ${games} games (w:${wins} t:${ties} c:${cutoffs})`);
