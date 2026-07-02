// Parity and tactics tests for the WASM engine against the JS domain rules.
// Skipped when the wasm artifact hasn't been built yet.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { newGame, applyMove, legalMoves, isLegal, idx, X, O } from '../js/domain/rules.js';
import { createWasmEngine } from '../js/ai/wasm.js';

const wasmPath = fileURLToPath(new URL('../wasm/engine.wasm', import.meta.url));
const bytes = await readFile(wasmPath).catch(() => null);
const skip = bytes ? false : 'wasm/engine.wasm not built';

test('wasm engine plays only JS-legal moves across random playouts', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { maxDepth: 2, nodeBudget: 20000, seed: 42 });
  let checked = 0;
  for (let g = 0; g < 60; g++) {
    let s = newGame();
    for (let ply = 0; ply < 40 && !s.result; ply++) {
      // Alternate: even plies random (drives state diversity), odd plies wasm.
      if (ply % 2 === 0) {
        const moves = legalMoves(s);
        s = applyMove(s, moves[(g * 7919 + ply * 104729) % moves.length]);
      } else {
        const m = await engine.chooseMove(s);
        assert.ok(isLegal(s, m), `illegal wasm move ${JSON.stringify(m)} at game ${g} ply ${ply}`);
        s = applyMove(s, m);
        checked++;
      }
    }
  }
  // Games end quickly (the engine beats random play fast); ~200 distinct
  // engine decisions is still broad coverage of the move generator.
  assert.ok(checked > 150, `too few positions exercised: ${checked}`);
});

test('wasm engine takes an immediate win', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { maxDepth: 4, nodeBudget: 200000, seed: 7 });
  const s = newGame();
  s.board[idx(1, 1)] = X; s.board[idx(1, 2)] = X;
  s.board[idx(3, 1)] = O; s.board[idx(3, 2)] = O;
  s.placed[X] = 2; s.placed[O] = 2;
  s.reserve[X] = 2; s.reserve[O] = 2;
  const move = await engine.chooseMove(s);
  const n = applyMove(s, move);
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, X);
});

test('wasm engine blocks an immediate loss', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { maxDepth: 4, nodeBudget: 200000, seed: 7 });
  const s = newGame();
  s.board[idx(1, 1)] = X; s.board[idx(1, 2)] = X;
  s.board[idx(3, 1)] = O; s.board[idx(2, 3)] = O;
  s.placed[X] = 2; s.placed[O] = 2;
  s.reserve[X] = 2; s.reserve[O] = 2;
  s.turn = O;
  const move = await engine.chooseMove(s);
  assert.ok(isLegal(s, move));
  const n = applyMove(s, move);
  assert.equal(n.result, null);
  const xWinsNext = legalMoves(n).some((m) => {
    const r = applyMove(n, m).result;
    return r?.type === 'win' && r.winner === X;
  });
  assert.equal(xWinsNext, false);
});

test('wasm engine wins with a grid slide when that is the only win', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { maxDepth: 4, nodeBudget: 200000, seed: 7 });
  const s = newGame();
  // X row on board row 3 cols 1-3; grid at (1,2) excludes it. Only the slide
  // down to (2,2) lights all three at once. X has no reserve.
  s.board[idx(3, 1)] = X; s.board[idx(3, 2)] = X; s.board[idx(3, 3)] = X; s.board[idx(0, 0)] = X;
  s.board[idx(0, 2)] = O; s.board[idx(0, 3)] = O; s.board[idx(4, 0)] = O; s.board[idx(4, 4)] = O;
  s.placed[X] = 4; s.placed[O] = 4;
  s.reserve[X] = 0; s.reserve[O] = 0;
  s.center = { r: 1, c: 2 };
  const move = await engine.chooseMove(s);
  assert.equal(move.type, 'grid');
  const n = applyMove(s, move);
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, X);
});

test('wasm engine wins with a piece move when placements are exhausted', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { maxDepth: 4, nodeBudget: 200000, seed: 7 });
  const s = newGame();
  // X: two on grid row 2, the third square empty; no reserve, no slide wins.
  s.board[idx(2, 1)] = X; s.board[idx(2, 2)] = X; s.board[idx(4, 4)] = X; s.board[idx(0, 0)] = X;
  s.board[idx(1, 1)] = O; s.board[idx(3, 3)] = O; s.board[idx(0, 4)] = O; s.board[idx(4, 0)] = O;
  s.placed[X] = 4; s.placed[O] = 4;
  s.reserve[X] = 0; s.reserve[O] = 0;
  const move = await engine.chooseMove(s);
  assert.equal(move.type, 'move');
  assert.equal(move.to, idx(2, 3));
  const n = applyMove(s, move);
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, X);
});

test('wasm engine rejects garbage input', { skip }, async () => {
  const engine = await createWasmEngine(bytes, { seed: 7 });
  const s = newGame();
  s.center = { r: 9, c: 9 };
  assert.equal(await engine.chooseMove(s), null);
});
