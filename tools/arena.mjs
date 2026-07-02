// Pits two engines against each other and reports W/L/T.
// Usage: node tools/arena.mjs <games> <engineA> <engineB>
//   engine spec: "js[:timeBudgetMs]" or "wasm:<path>[:maxDepth[:nodeBudget]]"

import { readFileSync } from 'node:fs';
import { newGame, applyMove, legalMoves, isLegal, X, O } from '../js/domain/rules.js';
import { createJsEngine } from '../js/ai/engine.js';
import { createWasmEngine } from '../js/ai/wasm.js';

async function makeEngine(spec, seed) {
  const [kind, ...rest] = spec.split(':');
  if (kind === 'js') return createJsEngine({ timeBudgetMs: Number(rest[0] ?? 650) });
  if (kind === 'wasm') {
    const bytes = readFileSync(rest[0]);
    return createWasmEngine(bytes, {
      maxDepth: Number(rest[1] ?? 6),
      nodeBudget: Number(rest[2] ?? 1_500_000),
      seed,
    });
  }
  throw new Error(`unknown engine spec: ${spec}`);
}

const games = Number(process.argv[2] ?? 20);
const specA = process.argv[3] ?? 'wasm:wasm/engine.wasm';
const specB = process.argv[4] ?? 'js:100';

let rngState = 20260702;
const rand = () => {
  rngState ^= rngState << 13; rngState >>>= 0;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5; rngState >>>= 0;
  return rngState / 0x100000000;
};

const tally = { A: 0, B: 0, tie: 0, cutoff: 0 };
for (let g = 0; g < games; g++) {
  const engineA = await makeEngine(specA, g * 2 + 1);
  const engineB = await makeEngine(specB, g * 2 + 2);
  const aPlays = g % 2 === 0 ? X : O; // alternate colors
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
  if (s.result?.type === 'win') tally[s.result.winner === aPlays ? 'A' : 'B']++;
  else if (s.result?.type === 'tie') tally.tie++;
  else tally.cutoff++;
  process.stdout.write('.');
}
console.log(`\nA=${specA} vs B=${specB} over ${games} games:`);
console.log(`  A wins: ${tally.A}  B wins: ${tally.B}  ties: ${tally.tie}  cutoffs: ${tally.cutoff}`);
