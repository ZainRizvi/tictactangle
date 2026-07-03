#!/bin/sh
# Builds a difficulty variant of the Rust engine to WebAssembly.
# Usage: wasm/build.sh <variant>   (a weights module at engine/weights/<variant>.rs)
# Drops the artifact where the site serves it: wasm/engine-<variant>.wasm
set -e
variant=${1:-medium}
cd "$(dirname "$0")/engine"
if [ ! -f "weights/$variant.rs" ]; then
  echo "no such weights variant: weights/$variant.rs" >&2
  exit 1
fi
cp "weights/$variant.rs" src/weights.rs
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/tictactwo_engine.wasm "../engine-$variant.wasm"
ls -la "../engine-$variant.wasm"
