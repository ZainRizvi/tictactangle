import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameSession } from '../js/app/session.js';
import { legalMoves, idx, X, O } from '../js/domain/rules.js';

const tick = () => new Promise((r) => setTimeout(r, 5));

test('human intents apply only when legal', () => {
  const session = new GameSession();
  assert.equal(session.place(idx(0, 0)), false); // outside the grid
  assert.equal(session.state.ply, 0);
  assert.equal(session.place(idx(2, 2)), true);
  assert.equal(session.state.board[idx(2, 2)], X);
  assert.equal(session.state.turn, O);
  assert.equal(session.slideGrid(1, 0), false); // locked until both placed two
});

test('subscribers get snapshots on every change', () => {
  const session = new GameSession();
  const seen = [];
  session.subscribe((s) => seen.push(s.state.ply));
  session.place(idx(2, 2));
  session.place(idx(1, 1));
  assert.deepEqual(seen, [0, 1, 2]);
});

test('an AI seat is pumped automatically and cancelled by reset', async () => {
  const scripted = {
    async chooseMove(state) {
      await tick();
      return legalMoves(state)[0];
    },
  };
  const session = new GameSession({ [X]: null, [O]: scripted });
  session.place(idx(2, 2)); // human X moves; O's controller should fire
  assert.equal(session.busySeat, O);
  assert.equal(session.canAct(), false);
  await tick(); await tick();
  assert.equal(session.state.ply, 2);
  assert.equal(session.state.turn, X);
  assert.equal(session.busySeat, null);
});

test('human intents are rejected while a controller is thinking', async () => {
  const slow = {
    chooseMove: (state) => new Promise((r) => setTimeout(() => r(legalMoves(state)[0]), 20)),
  };
  const session = new GameSession({ [X]: null, [O]: slow });
  session.place(idx(2, 2));
  assert.equal(session.busySeat, O);
  assert.equal(session.place(idx(1, 1)), false); // not X's turn to act
  assert.equal(session.state.ply, 1);
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(session.state.ply, 2);
});

test('a controller that returns an illegal move faults its seat', async () => {
  const bad = { async chooseMove() { return { type: 'grid', dr: 1, dc: 0 }; } }; // locked this early
  const session = new GameSession({ [X]: null, [O]: bad });
  session.place(idx(2, 2));
  await tick(); await tick();
  assert.equal(session.state.ply, 1); // O's move was rejected
  assert.equal(session.busySeat, null);
  assert.equal(session.seatFault, O);
  session.newGame();
  assert.equal(session.seatFault, null);
});

test('a controller that throws faults its seat', async () => {
  const broken = { async chooseMove() { throw new Error('boom'); } };
  const session = new GameSession({ [X]: null, [O]: broken });
  session.place(idx(2, 2));
  await tick(); await tick();
  assert.equal(session.seatFault, O);
  assert.equal(session.busySeat, null);
});

test('setSeats drops in-flight controller moves and clears faults', async () => {
  let resolveMove;
  const slow = { chooseMove: () => new Promise((r) => { resolveMove = r; }) };
  const session = new GameSession({ [X]: null, [O]: slow });
  session.place(idx(2, 2));
  assert.equal(session.busySeat, O);
  session.seatFault = O; // simulate a previously surfaced fault
  session.setSeats({ [X]: null, [O]: null }); // switch to PvP mid-think
  assert.equal(session.seatFault, null);
  resolveMove({ type: 'place', to: idx(2, 3) });
  await tick();
  assert.equal(session.state.ply, 0); // stale move never applied
  assert.equal(session.busySeat, null);
});

test('stale controller results are dropped after newGame', async () => {
  let resolveMove;
  const slow = { chooseMove: () => new Promise((r) => { resolveMove = r; }) };
  const session = new GameSession({ [X]: null, [O]: slow });
  session.place(idx(2, 2));
  assert.equal(session.busySeat, O);
  session.newGame();
  resolveMove({ type: 'place', to: idx(2, 3) });
  await tick();
  assert.equal(session.state.ply, 0); // stale move never applied
});
