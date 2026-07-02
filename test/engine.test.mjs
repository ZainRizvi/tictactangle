import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newGame, applyMove, legalMoves, isLegal, idx, X, O } from '../js/domain/rules.js';
import { createJsEngine } from '../js/ai/engine.js';

const engine = createJsEngine();

test('engine returns a legal move from the opening position', async () => {
  const s = newGame();
  const move = await engine.chooseMove(s);
  assert.ok(isLegal(s, move));
});

test('engine takes an immediate win', async () => {
  const s = newGame();
  // X to move can complete the grid's top row at (1,3).
  s.board[idx(1, 1)] = X; s.board[idx(1, 2)] = X;
  s.board[idx(3, 1)] = O; s.board[idx(3, 2)] = O;
  s.placed[X] = 2; s.placed[O] = 2;
  s.reserve[X] = 2; s.reserve[O] = 2;
  const move = await engine.chooseMove(s);
  const n = applyMove(s, move);
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, X);
});

test('engine blocks an immediate loss', async () => {
  const s = newGame();
  // X threatens (1,3); O to move has no win of its own and must prevent an
  // immediate X win (block, slide the grid away, or equivalent).
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
