// Engine adapter selection for the AI seat.

import { createJsEngine } from './engine.js';

/** @returns {Promise<import('../app/ports.js').PlayerController>} */
export async function createAiPlayer() {
  return createJsEngine();
}

/**
 * Paces a controller for play against a human: yields briefly so the UI can
 * paint its "thinking" state before a synchronous engine blocks the thread,
 * and never resolves faster than `ms` so the reply reads as a move, not a
 * glitch.
 */
export function withMinDelay(controller, ms) {
  return {
    async chooseMove(state) {
      const start = performance.now();
      await new Promise((r) => setTimeout(r, 30));
      const move = await controller.chooseMove(state);
      const remaining = ms - (performance.now() - start);
      if (remaining > 0) await new Promise((r) => setTimeout(r, remaining));
      return move;
    },
  };
}
