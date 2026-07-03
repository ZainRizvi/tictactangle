//! Inference for the RL policy+value network (the "impossible" difficulty).
//! Weights are the raw bytes of rl/models/best.bin (see rl/src/net.rs for
//! the layout), parsed once into static arrays. no_std: math helpers are
//! hand-rolled since core lacks float transcendentals.

use crate::State;

pub const FEATS: usize = 61;
pub const TRUNK: usize = 128;
pub const VHID: usize = 64;
pub const ACTIONS: usize = 658;

static RAW: &[u8] = include_bytes!("rlnet.bin");

static mut W1: [f32; FEATS * TRUNK] = [0.0; FEATS * TRUNK];
static mut B1: [f32; TRUNK] = [0.0; TRUNK];
static mut WV1: [f32; TRUNK * VHID] = [0.0; TRUNK * VHID];
static mut BV1: [f32; VHID] = [0.0; VHID];
static mut WV2: [f32; VHID] = [0.0; VHID];
static mut BV2: f32 = 0.0;
static mut WP: [f32; TRUNK * ACTIONS] = [0.0; TRUNK * ACTIONS];
static mut BP: [f32; ACTIONS] = [0.0; ACTIONS];
static mut READY: bool = false;

fn f32_at(off: usize) -> f32 {
    let b = [RAW[off], RAW[off + 1], RAW[off + 2], RAW[off + 3]];
    f32::from_le_bytes(b)
}

/// Parses the weight blob on first use. Single-threaded (wasm), so the
/// static-mut init flag is safe.
#[allow(static_mut_refs)]
pub fn ensure_ready() -> bool {
    unsafe {
        if READY {
            return true;
        }
        let need = (FEATS * TRUNK + TRUNK + TRUNK * VHID + VHID + VHID + 1 + TRUNK * ACTIONS + ACTIONS) * 4;
        if RAW.len() != need {
            return false; // placeholder blob — feature built without a model
        }
        let mut off = 0usize;
        let mut fill = |dst: &mut [f32], off: &mut usize| {
            for v in dst.iter_mut() {
                *v = f32_at(*off);
                *off += 4;
            }
        };
        fill(&mut W1, &mut off);
        fill(&mut B1, &mut off);
        fill(&mut WV1, &mut off);
        fill(&mut BV1, &mut off);
        fill(&mut WV2, &mut off);
        BV2 = f32_at(off);
        off += 4;
        fill(&mut WP, &mut off);
        fill(&mut BP, &mut off);
        READY = true;
        true
    }
}

#[inline]
fn fabs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

/// exp(x) via 2^(x/ln2): split into integer exponent + polynomial fraction.
/// Accurate to ~1e-6 relative over the softmax-relevant range.
pub fn expf(x: f32) -> f32 {
    let x = if x > 88.0 { 88.0 } else if x < -88.0 { -88.0 } else { x };
    let z = x * core::f32::consts::LOG2_E;
    let zi = if z >= 0.0 { (z + 0.5) as i32 } else { (z - 0.5) as i32 };
    let f = z - zi as f32; // in [-0.5, 0.5]
    // 2^f minimax polynomial
    let p = 0.000154_653_49_f32;
    let p = p * f + 0.001_339_35;
    let p = p * f + 0.009_618_37;
    let p = p * f + 0.055_503_27;
    let p = p * f + 0.240_226_51;
    let p = p * f + 0.693_147_18;
    let two_f = p * f + 1.0;
    f32::from_bits((((zi + 127) as u32) << 23)) * two_f
}

/// sqrt via bit-trick seed + two Newton steps (plenty for PUCT).
pub fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1FBD_1DF5);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

fn encode(s: &State, feat: &mut [f32; FEATS]) {
    for f in feat.iter_mut() {
        *f = 0.0;
    }
    let me = s.turn;
    let opp = 3 - me;
    let mut mine = 0i32;
    let mut theirs = 0i32;
    for i in 0..25 {
        if s.board[i] == me {
            feat[i] = 1.0;
            mine += 1;
        } else if s.board[i] == opp {
            feat[25 + i] = 1.0;
            theirs += 1;
        }
    }
    feat[50 + ((s.cr - 1) * 3 + (s.cc - 1)) as usize] = 1.0;
    feat[59] = (4 - mine) as f32 / 4.0;
    feat[60] = (4 - theirs) as f32 / 4.0;
}

/// Value in [-1,1] from the side-to-move perspective, plus priors over the
/// given legal action indices (written into `priors[..n]`).
#[allow(static_mut_refs)]
pub fn eval(s: &State, actions: &[usize], priors: &mut [f32]) -> f32 {
    let mut feat = [0.0f32; FEATS];
    encode(s, &mut feat);
    unsafe {
        let mut h = [0.0f32; TRUNK];
        h.copy_from_slice(&B1);
        for (i, &f) in feat.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            let row = &W1[i * TRUNK..(i + 1) * TRUNK];
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

        let mut hv = [0.0f32; VHID];
        hv.copy_from_slice(&BV1);
        for (i, &v) in h.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let row = &WV1[i * VHID..(i + 1) * VHID];
            for j in 0..VHID {
                hv[j] += row[j] * v;
            }
        }
        let mut vraw = BV2;
        for j in 0..VHID {
            if hv[j] > 0.0 {
                vraw += WV2[j] * hv[j];
            }
        }
        let value = vraw / (1.0 + fabs(vraw));

        let n = actions.len();
        let mut maxz = f32::NEG_INFINITY;
        for (k, &a) in actions.iter().enumerate() {
            let mut z = BP[a];
            for (j, &v) in h.iter().enumerate() {
                if v != 0.0 {
                    z += WP[j * ACTIONS + a] * v;
                }
            }
            priors[k] = z;
            if z > maxz {
                maxz = z;
            }
        }
        let mut sum = 0.0;
        for p in priors[..n].iter_mut() {
            *p = expf(*p - maxz);
            sum += *p;
        }
        for p in priors[..n].iter_mut() {
            *p /= sum;
        }
        value
    }
}
