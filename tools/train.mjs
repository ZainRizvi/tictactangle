// Trains the value network on self-play data. Pure JS (Float32Array + Adam).
//
// Architecture (must mirror wasm/engine/src/lib.rs eval_nn):
//   61 inputs -> 64 ReLU -> 32 ReLU -> 1 softsign (x / (1 + |x|))
// Inputs, from the side-to-move perspective:
//   0..25 own pieces, 25..50 opponent pieces, 50..59 grid-center one-hot,
//   59 own reserve/4, 60 opp reserve/4
//
// Usage: node tools/train.mjs "data/part-*.txt" [epochs]
// Writes wasm/engine/weights.json

import { readFileSync, writeFileSync, globSync } from 'node:fs';

const IN = 61, H1 = 64, H2 = 32;
const pattern = process.argv[2] ?? 'data/part-*.txt';
const EPOCHS = Number(process.argv[3] ?? 14);

// ---------- load data ----------

const files = globSync(pattern);
if (files.length === 0) throw new Error(`no data files match ${pattern}`);

const samples = [];
for (const f of files) {
  for (const line of readFileSync(f, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const [board, cr, cc, turn, z] = line.split(' ');
    samples.push({ board, cr: +cr, cc: +cc, turn: +turn, z: +z, file: f });
  }
}
console.log(`${samples.length} samples from ${files.length} files`);

function encode(s, out) {
  out.fill(0);
  const me = s.turn, opp = 3 - s.turn;
  let mine = 0, theirs = 0;
  for (let i = 0; i < 25; i++) {
    const v = +s.board[i];
    if (v === me) { out[i] = 1; mine++; }
    else if (v === opp) { out[25 + i] = 1; theirs++; }
  }
  out[50 + (s.cr - 1) * 3 + (s.cc - 1)] = 1;
  out[59] = (4 - mine) / 4;
  out[60] = (4 - theirs) / 4;
}

// ---------- model ----------

let seed = 1234567;
const rand = () => {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5; seed >>>= 0;
  return seed / 0x100000000;
};
const gauss = () => {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const heInit = (n, fanIn) => Float32Array.from({ length: n }, () => gauss() * Math.sqrt(2 / fanIn));
const params = {
  W1: heInit(IN * H1, IN), B1: new Float32Array(H1),
  W2: heInit(H1 * H2, H1), B2: new Float32Array(H2),
  W3: heInit(H2, H2), B3: new Float32Array(1),
};

// Adam state
const adam = {};
for (const k of Object.keys(params)) {
  adam[k] = { m: new Float32Array(params[k].length), v: new Float32Array(params[k].length) };
}
let adamT = 0;
const LR = 1e-3, BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;

function adamStep(key, grads) {
  const p = params[key], { m, v } = adam[key];
  const c1 = 1 - Math.pow(BETA1, adamT), c2 = 1 - Math.pow(BETA2, adamT);
  for (let i = 0; i < p.length; i++) {
    m[i] = BETA1 * m[i] + (1 - BETA1) * grads[i];
    v[i] = BETA2 * v[i] + (1 - BETA2) * grads[i] * grads[i];
    p[i] -= (LR * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + EPS);
  }
}

// ---------- training ----------

const BATCH = 128;
const x = new Float32Array(IN);
const h1 = new Float32Array(H1), h2 = new Float32Array(H2);
const d1 = new Float32Array(H1), d2 = new Float32Array(H2);
const gW1 = new Float32Array(IN * H1), gB1 = new Float32Array(H1);
const gW2 = new Float32Array(H1 * H2), gB2 = new Float32Array(H2);
const gW3 = new Float32Array(H2), gB3 = new Float32Array(1);

function forward() {
  const { W1, B1, W2, B2, W3, B3 } = params;
  for (let j = 0; j < H1; j++) {
    let a = B1[j];
    for (let i = 0; i < IN; i++) if (x[i] !== 0) a += W1[i * H1 + j] * x[i];
    h1[j] = a > 0 ? a : 0;
  }
  for (let j = 0; j < H2; j++) {
    let a = B2[j];
    for (let i = 0; i < H1; i++) a += W2[i * H2 + j] * h1[i];
    h2[j] = a > 0 ? a : 0;
  }
  let out = B3[0];
  for (let i = 0; i < H2; i++) out += W3[i] * h2[i];
  return out;
}

// Split. Positions within a game are highly correlated, so validation must
// be game-disjoint from training: hold out the whole last file (files are
// game-disjoint). With a single file we fall back to a position-level split
// and the validation metric is optimistic.
let val, train;
if (files.length > 1) {
  const valFile = files[files.length - 1];
  val = samples.filter((s) => s.file === valFile).slice(0, 6000);
  train = samples.filter((s) => s.file !== valFile);
} else {
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }
  const valN = Math.min(6000, Math.floor(samples.length * 0.05));
  val = samples.slice(0, valN);
  train = samples.slice(valN);
}
console.log(`train ${train.length}, val ${val.length} (game-disjoint: ${files.length > 1})`);
// initial shuffle of the training set
for (let i = train.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [train[i], train[j]] = [train[j], train[i]];
}

function evalVal() {
  let mse = 0, signOk = 0, signN = 0;
  for (const s of val) {
    encode(s, x);
    const raw = forward();
    const y = raw / (1 + Math.abs(raw));
    mse += (y - s.z) ** 2;
    if (s.z !== 0) { signN++; if (Math.sign(y) === Math.sign(s.z)) signOk++; }
  }
  return { mse: mse / val.length, acc: signOk / signN };
}

console.log('initial', evalVal());

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  const t0 = performance.now();
  // reshuffle train
  for (let i = train.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [train[i], train[j]] = [train[j], train[i]];
  }
  let trainMse = 0;
  for (let b = 0; b + BATCH <= train.length; b += BATCH) {
    gW1.fill(0); gB1.fill(0); gW2.fill(0); gB2.fill(0); gW3.fill(0); gB3.fill(0);
    for (let k = 0; k < BATCH; k++) {
      const s = train[b + k];
      encode(s, x);
      const raw = forward();
      const denom = 1 + Math.abs(raw);
      const y = raw / denom;
      const err = y - s.z;
      trainMse += err * err;
      // d(softsign)/draw = 1 / (1+|raw|)^2
      const dRaw = (2 * err) / (denom * denom) / BATCH;

      // output layer
      for (let i = 0; i < H2; i++) {
        gW3[i] += dRaw * h2[i];
        d2[i] = h2[i] > 0 ? dRaw * params.W3[i] : 0;
      }
      gB3[0] += dRaw;
      // hidden 2
      for (let i = 0; i < H1; i++) {
        let acc = 0;
        for (let j = 0; j < H2; j++) {
          if (d2[j] !== 0) { gW2[i * H2 + j] += d2[j] * h1[i]; acc += d2[j] * params.W2[i * H2 + j]; }
        }
        d1[i] = h1[i] > 0 ? acc : 0;
      }
      for (let j = 0; j < H2; j++) gB2[j] += d2[j];
      // hidden 1
      for (let i = 0; i < IN; i++) {
        if (x[i] === 0) continue;
        for (let j = 0; j < H1; j++) gW1[i * H1 + j] += d1[j] * x[i];
      }
      for (let j = 0; j < H1; j++) gB1[j] += d1[j];
    }
    adamT++;
    adamStep('W1', gW1); adamStep('B1', gB1);
    adamStep('W2', gW2); adamStep('B2', gB2);
    adamStep('W3', gW3); adamStep('B3', gB3);
  }
  const { mse, acc } = evalVal();
  console.log(
    `epoch ${epoch}: train_mse=${(trainMse / train.length).toFixed(4)} val_mse=${mse.toFixed(4)} val_sign_acc=${(acc * 100).toFixed(1)}% (${((performance.now() - t0) / 1000).toFixed(1)}s)`
  );
}

writeFileSync(
  'wasm/engine/weights.json',
  JSON.stringify({
    W1: [...params.W1], B1: [...params.B1],
    W2: [...params.W2], B2: [...params.B2],
    W3: [...params.W3], B3: params.B3[0],
  })
);
console.log('wrote wasm/engine/weights.json');
