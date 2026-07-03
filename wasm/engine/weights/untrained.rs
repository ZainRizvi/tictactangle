//! Placeholder weights: TRAINED=false makes the engine use the handcrafted
//! evaluation. Kept for differential testing of search vs evaluation.

pub const TRAINED: bool = false;
pub const H1: usize = 64;
pub const H2: usize = 32;

pub static W1: [f32; 61 * H1] = [0.0; 61 * H1];
pub static B1: [f32; H1] = [0.0; H1];
pub static W2: [f32; H1 * H2] = [0.0; H1 * H2];
pub static B2: [f32; H2] = [0.0; H2];
pub static W3: [f32; H2] = [0.0; H2];
pub static B3: f32 = 0.0;
