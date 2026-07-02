import { test } from 'node:test';
import assert from 'node:assert/strict';
import { withMinDelay } from '../js/ai/index.js';

test('withMinDelay passes the move through and enforces the floor', async () => {
  const move = { type: 'place', to: 12 };
  const instant = { async chooseMove() { return move; } };
  const paced = withMinDelay(instant, 120);
  const start = performance.now();
  const result = await paced.chooseMove({});
  const elapsed = performance.now() - start;
  assert.equal(result, move);
  assert.ok(elapsed >= 110, `resolved too fast: ${elapsed}ms`);
});

test('withMinDelay does not stack extra delay onto a slow controller', async () => {
  const slow = {
    chooseMove: () => new Promise((r) => setTimeout(() => r({ type: 'grid', dr: 1, dc: 0 }), 100)),
  };
  const paced = withMinDelay(slow, 50);
  const start = performance.now();
  await paced.chooseMove({});
  const elapsed = performance.now() - start;
  assert.ok(elapsed < 200, `took too long: ${elapsed}ms`);
});
