#!/bin/sh
# Builds the Rust engine to WebAssembly and drops the artifact where the
# site serves it (wasm/engine.wasm).
set -e
cd "$(dirname "$0")/engine"
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/tictactwo_engine.wasm ../engine.wasm
ls -la ../engine.wasm
