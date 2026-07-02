// AI adapter selection. Prefers the WASM neural model; falls back to the
// pure-JS engine if WASM is unavailable in this browser.

import { createJsEngine } from './engine.js';

/** @returns {Promise<import('../app/ports.js').PlayerController>} */
export async function createAiPlayer() {
  return createJsEngine();
}

/**
 * Wraps a controller so it never resolves faster than `ms` — pacing so the
 * opponent's reply reads as a move, not a glitch.
 */
export function withMinDelay(controller, ms) {
  return {
    async chooseMove(state) {
      const [move] = await Promise.all([
        controller.chooseMove(state),
        new Promise((r) => setTimeout(r, ms)),
      ]);
      return move;
    },
  };
}
