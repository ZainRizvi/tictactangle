// Adapter for the WASM engine (Rust negamax + embedded value network).
// Implements the PlayerController port. Platform-neutral: the caller supplies
// the wasm bytes (fetch in the browser, fs in Node tests).

const NO_MOVE = 0xffffffff;

/**
 * @param {BufferSource|WebAssembly.Module} source compiled or raw wasm
 * @param {{maxDepth?: number, nodeBudget?: number, seed?: number}} [opts]
 * @returns {Promise<import('../app/ports.js').PlayerController>}
 */
export async function createWasmEngine(source, opts = {}) {
  // instantiate() returns {module, instance} for bytes but a bare Instance
  // for a precompiled Module — normalize both shapes.
  const result = await WebAssembly.instantiate(source, {});
  const instance = result.instance ?? result;
  const { input_ptr, choose_move, set_seed, memory } = instance.exports;
  const ptr = input_ptr();
  set_seed((opts.seed ?? (Math.random() * 0xffffffff)) >>> 0);
  const maxDepth = opts.maxDepth ?? 6;
  const nodeBudget = opts.nodeBudget ?? 1_500_000;

  return {
    async chooseMove(state) {
      // The memory view must be rebuilt per call: growth detaches buffers.
      const buf = new Uint8Array(memory.buffer, ptr, 32);
      for (let i = 0; i < 25; i++) buf[i] = state.board[i];
      buf[25] = state.center.r;
      buf[26] = state.center.c;
      buf[27] = state.turn;
      buf[28] = state.bannedCenter?.r ?? 0;
      buf[29] = state.bannedCenter?.c ?? 0;
      const packed = choose_move(maxDepth, nodeBudget) >>> 0;
      if (packed === NO_MOVE) return null;
      const kind = packed >> 16;
      const a = (packed >> 8) & 0xff;
      const b = packed & 0xff;
      if (kind === 0) return { type: 'place', to: a };
      if (kind === 1) return { type: 'grid', dr: a - 1, dc: b - 1 };
      return { type: 'move', from: a, to: b };
    },
  };
}
