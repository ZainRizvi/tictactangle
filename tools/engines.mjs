// Shared engine-spec parser for the tools.
//   "js[:timeBudgetMs]"                      pure-JS engine
//   "wasm:<path>[:maxDepth[:nodeBudget]]"    WASM engine variant
import { readFileSync } from 'node:fs';
import { createJsEngine } from '../js/ai/engine.js';
import { createWasmEngine } from '../js/ai/wasm.js';

export async function makeEngine(spec, seed) {
  const [kind, ...rest] = spec.split(':');
  if (kind === 'js') return createJsEngine({ timeBudgetMs: Number(rest[0] ?? 650) });
  if (kind === 'wasm') {
    const bytes = readFileSync(rest[0]);
    return createWasmEngine(bytes, {
      maxDepth: Number(rest[1] ?? 6),
      nodeBudget: Number(rest[2] ?? 1_500_000),
      seed,
    });
  }
  throw new Error(`unknown engine spec: ${spec}`);
}
