// Pits two engines against each other and reports W/L/T.
// Usage: node tools/arena.mjs <games> <engineA> <engineB> [aColor]
//   engine spec: "js[:timeBudgetMs]" or "wasm:<path>[:maxDepth[:nodeBudget]]"
//   aColor: "alt" (default), "X", or "O" — fix engine A's color

import { newGame, applyMove, legalMoves, isLegal, X, O } from '../js/domain/rules.js';
import { makeEngine } from './engines.mjs';

const games = Number(process.argv[2] ?? 20);
const specA = process.argv[3] ?? 'wasm:wasm/engine-medium.wasm';
const specB = process.argv[4] ?? 'js:100';
const aColor = process.argv[5] ?? 'alt';

let rngState = 20260702;
const rand = () => {
  rngState ^= rngState << 13; rngState >>>= 0;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5; rngState >>>= 0;
  return rngState / 0x100000000;
};

const tally = { A: 0, B: 0, tie: 0, cutoff: 0, AasX: 0, AasO: 0, BasX: 0, BasO: 0 };
const aWinPlies = [];
const aLossPlies = [];
for (let g = 0; g < games; g++) {
  const engineA = await makeEngine(specA, g * 2 + 1);
  const engineB = await makeEngine(specB, g * 2 + 2);
  const aPlays = aColor === 'X' ? X : aColor === 'O' ? O : g % 2 === 0 ? X : O;
  let s = newGame();
  // two random opening plies for variety
  for (let i = 0; i < 2; i++) {
    const moves = legalMoves(s);
    s = applyMove(s, moves[Math.floor(rand() * moves.length)]);
  }
  while (!s.result && s.ply < 100) {
    const engine = s.turn === aPlays ? engineA : engineB;
    const m = await engine.chooseMove(s);
    if (!isLegal(s, m)) throw new Error(`illegal move from ${s.turn === aPlays ? specA : specB}: ${JSON.stringify(m)}`);
    s = applyMove(s, m);
  }
  if (s.result?.type === 'win') {
    const aWon = s.result.winner === aPlays;
    tally[aWon ? 'A' : 'B']++;
    if (aWon) {
      tally[aPlays === X ? 'AasX' : 'AasO']++;
      aWinPlies.push(s.ply);
    } else {
      tally[aPlays === X ? 'BasO' : 'BasX']++;
      aLossPlies.push(s.ply);
    }
  } else if (s.result?.type === 'tie') tally.tie++;
  else tally.cutoff++;
  process.stdout.write('.');
}
const avg = (xs) => (xs.length ? (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(1) : '-');
console.log(`\nA=${specA} vs B=${specB} over ${games} games (A color: ${aColor}):`);
console.log(`  A wins: ${tally.A} (as X: ${tally.AasX}, as O: ${tally.AasO})  B wins: ${tally.B} (as X: ${tally.BasX}, as O: ${tally.BasO})  ties: ${tally.tie}  cutoffs: ${tally.cutoff}`);
console.log(`  avg plies when A wins: ${avg(aWinPlies)}  when A loses: ${avg(aLossPlies)}`);
