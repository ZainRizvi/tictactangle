// Engine adapter selection for the AI seat: the WASM engine (Rust negamax
// with an embedded value network) when it loads, else the pure-JS engine.

import { createJsEngine } from './engine.js';
import { createWasmEngine } from './wasm.js';

/** @returns {Promise<import('../app/ports.js').PlayerController>} */
export async function createAiPlayer() {
  try {
    const res = await fetch(new URL('../../wasm/engine.wasm', import.meta.url));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await createWasmEngine(await res.arrayBuffer());
  } catch (err) {
    console.warn('WASM engine unavailable, using JS engine', err);
    return createJsEngine();
  }
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
