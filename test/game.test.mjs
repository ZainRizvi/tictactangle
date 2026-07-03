import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  newGame, applyMove, legalMoves, actionsUnlocked, findLine, isLegal,
  gridCells, idx, X, O, EMPTY,
} from '../js/domain/rules.js';

test('initial state: only placements inside centered grid', () => {
  const s = newGame();
  const moves = legalMoves(s);
  assert.equal(moves.length, 9);
  assert.ok(moves.every((m) => m.type === 'place'));
  assert.deepEqual(gridCells(s).sort((a, b) => a - b), [6, 7, 8, 11, 12, 13, 16, 17, 18]);
});

test('grid/piece moves unlock after each player places two', () => {
  let s = newGame();
  assert.ok(!actionsUnlocked(s));
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // O
  assert.ok(!actionsUnlocked(s));
  assert.ok(legalMoves(s).every((m) => m.type === 'place'));
  s = applyMove(s, { type: 'place', to: idx(2, 1) }); // X
  assert.ok(!actionsUnlocked(s));
  s = applyMove(s, { type: 'place', to: idx(2, 2) }); // O
  assert.ok(actionsUnlocked(s));
  const types = new Set(legalMoves(s).map((m) => m.type));
  assert.deepEqual([...types].sort(), ['grid', 'move', 'place']);
});

test('placement win: three in a row inside grid', () => {
  let s = newGame();
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(2, 2) }); // O
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // X
  s = applyMove(s, { type: 'place', to: idx(2, 3) }); // O
  s = applyMove(s, { type: 'place', to: idx(1, 3) }); // X wins top row of grid
  assert.equal(s.result?.type, 'win');
  assert.equal(s.result.winner, X);
  assert.deepEqual(s.result.line.sort((a, b) => a - b), [idx(1, 1), idx(1, 2), idx(1, 3)]);
});

test('pieces outside grid do not count toward a win', () => {
  let s = newGame();
  // X builds a row at board row 1 (cols 1-3), then grid slides down away from it.
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(3, 1) }); // O
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // X
  s = applyMove(s, { type: 'place', to: idx(3, 2) }); // O
  s = applyMove(s, { type: 'grid', dr: 1, dc: 0 }); // X slides grid down; center (3,2)
  assert.equal(s.result, null);
  // X row at board row 1 is now outside the grid: no winner.
  assert.equal(findLine(s, X), null);
});

test('grid slide can produce a win', () => {
  const s = newGame();
  // X row on board row 3 cols 1-3; grid centered at (1,2) (rows 0-2) excludes it.
  s.board[idx(3, 1)] = X; s.board[idx(3, 2)] = X; s.board[idx(3, 3)] = X;
  s.board[idx(0, 1)] = O; s.board[idx(0, 2)] = O;
  s.placed[X] = 3; s.placed[O] = 2;
  s.reserve[X] = 1; s.reserve[O] = 2;
  s.center = { r: 1, c: 2 };
  assert.equal(findLine(s, X), null);
  // X slides the grid down: center (2,2), grid rows 1-3 → X's row is inside.
  const n = applyMove(s, { type: 'grid', dr: 1, dc: 0 });
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, X);
});

test('grid slide revealing lines for both players is a tie', () => {
  const s = newGame();
  // X row on board row 2 cols 2-4, O row on board row 4 cols 2-4. With the
  // grid centered at (2,2) neither row is fully lit; sliding to (3,3) lights
  // both at once.
  s.board[idx(2, 2)] = X; s.board[idx(2, 3)] = X; s.board[idx(2, 4)] = X;
  s.board[idx(4, 2)] = O; s.board[idx(4, 3)] = O; s.board[idx(4, 4)] = O;
  s.placed[X] = 3; s.placed[O] = 3;
  s.reserve[X] = 1; s.reserve[O] = 1;
  const n = applyMove(s, { type: 'grid', dr: 1, dc: 1 });
  assert.equal(n.result?.type, 'tie');
});

test('sliding the grid onto only the opponent line loses for the mover', () => {
  const s = newGame();
  // O has a row on board row 3 cols 1-3; grid centered at (1,2) excludes it.
  // X (to move) slides the grid down and reveals it: O wins.
  s.board[idx(3, 1)] = O; s.board[idx(3, 2)] = O; s.board[idx(3, 3)] = O;
  s.board[idx(0, 1)] = X; s.board[idx(0, 2)] = X; s.board[idx(1, 1)] = X;
  s.placed[X] = 3; s.placed[O] = 3;
  s.reserve[X] = 1; s.reserve[O] = 1;
  s.center = { r: 1, c: 2 };
  const n = applyMove(s, { type: 'grid', dr: 1, dc: 0 });
  assert.equal(n.result?.type, 'win');
  assert.equal(n.result.winner, O);
});

test('piece move: from anywhere on board to empty grid cell', () => {
  let s = newGame();
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // O
  s = applyMove(s, { type: 'place', to: idx(2, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(2, 2) }); // O
  s = applyMove(s, { type: 'grid', dr: 1, dc: 1 }); // X: grid now rows 2..4 cols 2..4
  // O's piece at (1,2) is outside the grid; O may move it into the grid.
  const moves = legalMoves(s);
  const mv = moves.find((m) => m.type === 'move' && m.from === idx(1, 2));
  assert.ok(mv);
  s = applyMove(s, mv);
  assert.equal(s.board[idx(1, 2)], EMPTY);
  assert.equal(s.board[mv.to], O);
});

test('reserve exhaustion: no placements once 4 pieces are down', () => {
  let s = newGame();
  const placeAt = (cells) => { for (const c of cells) s = applyMove(s, { type: 'place', to: c }); };
  // Alternate X,O placements chosen so neither side ever has 3-in-a-row.
  placeAt([idx(1, 1), idx(1, 3), idx(1, 2), idx(2, 1), idx(2, 3), idx(2, 2), idx(3, 1), idx(3, 3)]);
  assert.equal(s.reserve[X], 0);
  assert.equal(s.reserve[O], 0);
  assert.equal(s.result, null);
  assert.ok(legalMoves(s).every((m) => m.type !== 'place'));
});

test('isLegal validates per move type, tolerating serialized null fields', () => {
  let s = newGame();
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // O
  s = applyMove(s, { type: 'place', to: idx(2, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(2, 2) }); // O — actions now unlocked
  // A JSON round-trip turns absent fields into nulls; still legal.
  assert.ok(isLegal(s, { type: 'place', to: idx(3, 3), from: null, dr: null, dc: null }));
  assert.ok(isLegal(s, { type: 'grid', dr: 1, dc: 0, from: null, to: null }));
  // Moving a piece to a cell outside the grid is rejected.
  assert.ok(isLegal(s, { type: 'move', from: idx(1, 1), to: idx(3, 3) }));
  assert.ok(!isLegal(s, { type: 'move', from: idx(1, 1), to: idx(0, 0) }));
  // Moving the opponent's piece is rejected.
  assert.ok(!isLegal(s, { type: 'move', from: idx(1, 2), to: idx(3, 3) }));
  assert.ok(!isLegal(s, null));
});

test('the reply may slide the grid straight back (official rules)', () => {
  let s = newGame();
  s = applyMove(s, { type: 'place', to: idx(1, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(1, 2) }); // O
  s = applyMove(s, { type: 'place', to: idx(2, 1) }); // X
  s = applyMove(s, { type: 'place', to: idx(2, 2) }); // O
  s = applyMove(s, { type: 'grid', dr: 1, dc: 0 }); // X slides (2,2) → (3,2)
  assert.ok(isLegal(s, { type: 'grid', dr: -1, dc: 0 })); // O may revert
});

test('grid cannot slide off the board', () => {
  let s = newGame();
  s = applyMove(s, { type: 'place', to: idx(1, 1) });
  s = applyMove(s, { type: 'place', to: idx(1, 2) });
  s = applyMove(s, { type: 'place', to: idx(2, 1) });
  s = applyMove(s, { type: 'place', to: idx(2, 2) });
  s = applyMove(s, { type: 'grid', dr: -1, dc: -1 }); // X: center now (1,1) — top-left extreme
  const gridMoves = legalMoves(s).filter((m) => m.type === 'grid');
  assert.ok(gridMoves.every((m) => m.dr >= 0 && m.dc >= 0));
  assert.equal(gridMoves.length, 3);
});
