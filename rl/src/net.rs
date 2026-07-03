//! Policy+value network with hand-rolled backprop and Adam.
//!
//! Architecture:
//!   61 features -> 128 ReLU trunk
//!     value head:  128 -> 64 ReLU -> 1 softsign  (predicted outcome, [-1,1])
//!     policy head: 128 -> 658 logits (masked softmax over legal actions)

use crate::game::{ACTIONS, FEATS};
use std::fs;
use std::io;
use std::path::Path;

pub const TRUNK: usize = 128;
pub const VHID: usize = 64;

pub struct Net {
    pub w1: Vec<f32>, // FEATS x TRUNK
    pub b1: Vec<f32>, // TRUNK
    pub wv1: Vec<f32>, // TRUNK x VHID
    pub bv1: Vec<f32>, // VHID
    pub wv2: Vec<f32>, // VHID
    pub bv2: f32,
    pub wp: Vec<f32>, // TRUNK x ACTIONS
    pub bp: Vec<f32>, // ACTIONS
}

pub struct Eval {
    pub value: f32,
    /// Prior probability per legal move (parallel to the caller's move list).
    pub priors: Vec<f32>,
}

fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

impl Net {
    pub fn new(rng: &mut crate::rng::Rng) -> Net {
        let he = |n: usize, fan_in: usize, rng: &mut crate::rng::Rng| -> Vec<f32> {
            let s = (2.0 / fan_in as f32).sqrt();
            (0..n).map(|_| rng.gauss() * s).collect()
        };
        Net {
            w1: he(FEATS * TRUNK, FEATS, rng),
            b1: vec![0.0; TRUNK],
            wv1: he(TRUNK * VHID, TRUNK, rng),
            bv1: vec![0.0; VHID],
            wv2: he(VHID, VHID, rng),
            bv2: 0.0,
            wp: he(TRUNK * ACTIONS, TRUNK, rng),
            bp: vec![0.0; ACTIONS],
        }
    }

    /// Trunk activations for a feature vector (sparse input).
    pub fn trunk(&self, feat: &[f32; FEATS]) -> Vec<f32> {
        let mut h = self.b1.clone();
        for (i, &f) in feat.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            let row = &self.w1[i * TRUNK..(i + 1) * TRUNK];
            if f == 1.0 {
                for j in 0..TRUNK {
                    h[j] += row[j];
                }
            } else {
                for j in 0..TRUNK {
                    h[j] += row[j] * f;
                }
            }
        }
        for v in h.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        h
    }

    pub fn value_from_trunk(&self, h: &[f32]) -> f32 {
        let mut hv = self.bv1.clone();
        for (i, &v) in h.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let row = &self.wv1[i * VHID..(i + 1) * VHID];
            for j in 0..VHID {
                hv[j] += row[j] * v;
            }
        }
        let mut out = self.bv2;
        for j in 0..VHID {
            if hv[j] > 0.0 {
                out += self.wv2[j] * hv[j];
            }
        }
        softsign(out)
    }

    /// Value + priors over the given legal action indices.
    /// Only legal logits are computed (the rest never matter).
    pub fn eval(&self, feat: &[f32; FEATS], legal_actions: &[usize]) -> Eval {
        let h = self.trunk(feat);
        let value = self.value_from_trunk(&h);

        let mut logits: Vec<f32> = legal_actions
            .iter()
            .map(|&a| {
                let mut z = self.bp[a];
                for (j, &v) in h.iter().enumerate() {
                    if v != 0.0 {
                        z += self.wp[j * ACTIONS + a] * v;
                    }
                }
                z
            })
            .collect();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for z in logits.iter_mut() {
            *z = (*z - max).exp();
            sum += *z;
        }
        for z in logits.iter_mut() {
            *z /= sum;
        }
        Eval { value, priors: logits }
    }

    pub fn params(&mut self) -> Vec<&mut Vec<f32>> {
        vec![
            &mut self.w1, &mut self.b1,
            &mut self.wv1, &mut self.bv1, &mut self.wv2,
            &mut self.wp, &mut self.bp,
        ]
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut bytes: Vec<u8> = Vec::new();
        let push = |v: &[f32], bytes: &mut Vec<u8>| {
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        };
        push(&self.w1, &mut bytes);
        push(&self.b1, &mut bytes);
        push(&self.wv1, &mut bytes);
        push(&self.bv1, &mut bytes);
        push(&self.wv2, &mut bytes);
        push(&[self.bv2], &mut bytes);
        push(&self.wp, &mut bytes);
        push(&self.bp, &mut bytes);
        fs::write(path, bytes)
    }

    pub fn load(path: &Path) -> io::Result<Net> {
        let bytes = fs::read(path)?;
        let mut off = 0usize;
        let take = |n: usize, off: &mut usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[*off..*off + 4]);
                v.push(f32::from_le_bytes(b));
                *off += 4;
            }
            v
        };
        let w1 = take(FEATS * TRUNK, &mut off);
        let b1 = take(TRUNK, &mut off);
        let wv1 = take(TRUNK * VHID, &mut off);
        let bv1 = take(VHID, &mut off);
        let wv2 = take(VHID, &mut off);
        let bv2 = take(1, &mut off)[0];
        let wp = take(TRUNK * ACTIONS, &mut off);
        let bp = take(ACTIONS, &mut off);
        assert_eq!(off, bytes.len(), "model file size mismatch");
        Ok(Net { w1, b1, wv1, bv1, wv2, bv2, wp, bp })
    }

    pub fn clone_net(&self) -> Net {
        Net {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            wv1: self.wv1.clone(),
            bv1: self.bv1.clone(),
            wv2: self.wv2.clone(),
            bv2: self.bv2,
            wp: self.wp.clone(),
            bp: self.bp.clone(),
        }
    }
}

/// One training sample.
pub struct Sample {
    pub feat: [f32; FEATS],
    pub actions: Vec<usize>, // legal action indices
    pub pi: Vec<f32>,        // MCTS visit distribution over those actions
    pub z: f32,              // final outcome from side-to-move perspective
}

/// Adam optimizer state + gradient buffers, mirroring Net's parameter layout.
pub struct Trainer {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    g: Vec<Vec<f32>>,
    t: i32,
    pub lr: f32,
}

const B1_: f32 = 0.9;
const B2_: f32 = 0.999;
const EPS: f32 = 1e-8;

impl Trainer {
    pub fn new(net: &mut Net, lr: f32) -> Trainer {
        let sizes: Vec<usize> = net.params().iter().map(|p| p.len()).collect();
        Trainer {
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            g: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            t: 0,
            lr,
        }
    }

    /// Accumulates gradients for one sample (call `step` after a batch).
    /// Returns (value_loss, policy_loss). NOTE: bv2's gradient rides in g[?]
    /// — bv2 is a scalar outside params(); handled explicitly below.
    pub fn accumulate(&mut self, net: &Net, s: &Sample, bv2_grad: &mut f32, scale: f32) -> (f32, f32) {
        // ---- forward, keeping intermediates ----
        let h = net.trunk(&s.feat);
        let mut hv = net.bv1.clone();
        for (i, &v) in h.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let row = &net.wv1[i * VHID..(i + 1) * VHID];
            for j in 0..VHID {
                hv[j] += row[j] * v;
            }
        }
        let mut vraw = net.bv2;
        for j in 0..VHID {
            if hv[j] > 0.0 {
                vraw += net.wv2[j] * hv[j];
            }
        }
        let value = softsign(vraw);

        let logits: Vec<f32> = s
            .actions
            .iter()
            .map(|&a| {
                let mut z = net.bp[a];
                for (j, &v) in h.iter().enumerate() {
                    if v != 0.0 {
                        z += net.wp[j * ACTIONS + a] * v;
                    }
                }
                z
            })
            .collect();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        let exps: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
        for &e in &exps {
            sum += e;
        }
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let vloss = (value - s.z) * (value - s.z);
        let mut ploss = 0.0;
        for (k, &p) in probs.iter().enumerate() {
            if s.pi[k] > 0.0 {
                ploss -= s.pi[k] * p.max(1e-9).ln();
            }
        }

        // ---- backward ----
        // dL/dh accumulates from both heads.
        let mut dh = vec![0.0f32; TRUNK];

        // value head: dL/dvraw = 2(value - z) * softsign'(vraw)
        let denom = 1.0 + vraw.abs();
        let dvraw = 2.0 * (value - s.z) / (denom * denom) * scale;
        *bv2_grad += dvraw;
        // indices in params(): 0 w1, 1 b1, 2 wv1, 3 bv1, 4 wv2, 5 wp, 6 bp
        for j in 0..VHID {
            if hv[j] > 0.0 {
                self.g[4][j] += dvraw * hv[j];
                let dhv = dvraw * net.wv2[j];
                self.g[3][j] += dhv;
                for (i, &v) in h.iter().enumerate() {
                    if v != 0.0 {
                        self.g[2][i * VHID + j] += dhv * v;
                        dh[i] += dhv * net.wv1[i * VHID + j];
                    }
                }
            }
        }

        // policy head: dL/dlogit_k = (p_k - pi_k)
        for (k, &a) in s.actions.iter().enumerate() {
            let dz = (probs[k] - s.pi[k]) * scale;
            if dz == 0.0 {
                continue;
            }
            self.g[6][a] += dz;
            for (j, &v) in h.iter().enumerate() {
                if v != 0.0 {
                    self.g[5][j * ACTIONS + a] += dz * v;
                    dh[j] += dz * net.wp[j * ACTIONS + a];
                }
            }
        }

        // trunk: ReLU mask then sparse input rows
        for j in 0..TRUNK {
            if h[j] == 0.0 {
                dh[j] = 0.0;
            }
        }
        for j in 0..TRUNK {
            if dh[j] != 0.0 {
                self.g[1][j] += dh[j];
            }
        }
        for (i, &f) in s.feat.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            for j in 0..TRUNK {
                if dh[j] != 0.0 {
                    self.g[0][i * TRUNK + j] += dh[j] * f;
                }
            }
        }

        (vloss, ploss)
    }

    pub fn step(&mut self, net: &mut Net, bv2_grad: f32) {
        self.t += 1;
        let c1 = 1.0 - B1_.powi(self.t);
        let c2 = 1.0 - B2_.powi(self.t);
        let lr = self.lr;
        {
            let params = net.params();
            for (pi, p) in params.into_iter().enumerate() {
                let (m, v, g) = (&mut self.m[pi], &mut self.v[pi], &mut self.g[pi]);
                for i in 0..p.len() {
                    if g[i] == 0.0 && m[i] == 0.0 && v[i] == 0.0 {
                        continue;
                    }
                    m[i] = B1_ * m[i] + (1.0 - B1_) * g[i];
                    v[i] = B2_ * v[i] + (1.0 - B2_) * g[i] * g[i];
                    p[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + EPS);
                    g[i] = 0.0;
                }
            }
        }
        // bv2 handled with plain SGD (single scalar; Adam overkill)
        net.bv2 -= lr * bv2_grad;
    }
}
