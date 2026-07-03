#!/bin/sh
# Builds a difficulty variant of the Rust engine to WebAssembly.
# Usage: wasm/build.sh <variant>
#   medium|hard   alpha-beta engine with the variant's value net
#                 (weights module at engine/weights/<variant>.rs)
#   impossible    MCTS over the RL policy/value net (embeds rl/models/best.bin)
# Drops the artifact where the site serves it: wasm/engine-<variant>.wasm
set -e
variant=${1:-medium}
cd "$(dirname "$0")/engine"
if [ "$variant" = "impossible" ]; then
  if [ -f ../../rl/models/best.bin ]; then
    cp ../../rl/models/best.bin src/rlnet.bin
  elif [ ! -f src/rlnet.bin ]; then
    echo "no RL model: neither rl/models/best.bin nor src/rlnet.bin exists" >&2
    exit 1
  fi
  # the alpha-beta weights module must still compile; reuse hard's
  cp weights/hard.rs src/weights.rs
  cargo build --release --target wasm32-unknown-unknown --features rl
else
  if [ ! -f "weights/$variant.rs" ]; then
    echo "no such weights variant: weights/$variant.rs" >&2
    exit 1
  fi
  cp "weights/$variant.rs" src/weights.rs
  cargo build --release --target wasm32-unknown-unknown
fi
cp target/wasm32-unknown-unknown/release/tictactwo_engine.wasm "../engine-$variant.wasm"
ls -la "../engine-$variant.wasm"
