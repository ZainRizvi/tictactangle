//! Small deterministic RNG (xorshift64) with gaussian, uniform, and
//! Dirichlet sampling — no external crates. Only high bits are consumed,
//! which is adequate quality for exploration noise and shuffles.

pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { s: seed | 1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        self.s
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn gauss(&mut self) -> f32 {
        // Box-Muller
        let mut u = 0.0;
        while u == 0.0 {
            u = self.uniform();
        }
        let v = self.uniform();
        (-2.0 * (u as f64).ln()).sqrt() as f32 * (2.0 * std::f32::consts::PI * v).cos()
    }

    /// Gamma(alpha, 1) via Marsaglia-Tsang (with the alpha<1 boost).
    pub fn gamma(&mut self, alpha: f32) -> f32 {
        if alpha < 1.0 {
            let g = self.gamma(alpha + 1.0);
            let mut u = 0.0;
            while u == 0.0 {
                u = self.uniform();
            }
            return g * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.gauss();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.uniform().max(1e-12);
            if u.ln() < 0.5 * x * x + d - d * v3 + d * v3.ln() {
                return d * v3;
            }
        }
    }

    /// Symmetric Dirichlet(alpha) of dimension n.
    pub fn dirichlet(&mut self, alpha: f32, n: usize) -> Vec<f32> {
        let mut xs: Vec<f32> = (0..n).map(|_| self.gamma(alpha).max(1e-10)).collect();
        let sum: f32 = xs.iter().sum();
        for x in xs.iter_mut() {
            *x /= sum;
        }
        xs
    }
}
