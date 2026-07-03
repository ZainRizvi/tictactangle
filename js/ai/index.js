// Engine adapter selection for the AI seat: a WASM engine variant (Rust
// negamax with an embedded value network) chosen by difficulty, or the
// pure-JS engine when WASM is unavailable.

import { createJsEngine } from './engine.js';
import { createWasmEngine } from './wasm.js';

export const DIFFICULTIES = {
  medium: { file: 'engine-medium.wasm', maxDepth: 6, nodeBudget: 1_500_000 },
  hard: { file: 'engine-hard.wasm', maxDepth: 8, nodeBudget: 600_000 },
  // AlphaZero-style: MCTS over the self-play-trained policy/value net.
  impossible: { file: 'engine-impossible.wasm', mcts: 8000 },
};

/**
 * @param {'medium'|'hard'} [difficulty]
 * @returns {Promise<import('../app/ports.js').PlayerController>}
 */
export async function createAiPlayer(difficulty = 'medium') {
  const cfg = DIFFICULTIES[difficulty] ?? DIFFICULTIES.medium;
  try {
    const res = await fetch(new URL(`../../wasm/${cfg.file}`, import.meta.url));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await createWasmEngine(await res.arrayBuffer(), {
      maxDepth: cfg.maxDepth,
      nodeBudget: cfg.nodeBudget,
      mcts: cfg.mcts,
    });
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
