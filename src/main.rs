use std::f32::consts::PI;

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

/// Generate a ±1 PRN code using ChaCha12Rng.
/// The seed can be any byte slice; it will be zero-padded or truncated to 32 bytes (256 bits).
pub fn generate_prn_chacha12(length: usize, seed: &[u8]) -> Vec<i8> {
    let mut key = [0u8; 32];

    // Copy user seed into key (truncate or pad)
    let n = seed.len().min(32);
    key[..n].copy_from_slice(&seed[..n]);

    let mut rng = ChaCha12Rng::from_seed(key);

    let mut prn = Vec::with_capacity(length);
    for _ in 0..length {
        // Draw a random bit from ChaCha12
        let r = (rng.next_u32() & 1) as i32;
        prn.push(if r == 0 { -1 } else { 1 });
    }
    prn
}

/// Configuration for the DSSS encoder/decoder.
#[derive(Clone, Debug)]
pub struct DsssConfig {
    /// Chips per information bit (processing gain).
    pub spreading_factor: usize,

    /// Chip rate in chips per second (sets effective bandwidth).
    pub chip_rate: f32,

    /// Carrier frequency in Hz for passband modulation.
    pub carrier_freq: f32,

    /// How many samples to generate per chip for passband.
    pub samples_per_chip: usize,

    /// The PRN spreading code (+1/-1 values). This defines the logical "channel".
    /// Length can be <= spreading_factor; it will be repeated as needed.
    pub prn_code: Vec<i8>,
}

impl DsssConfig {
    /// Derived sample rate in samples per second for passband.
    pub fn sample_rate(&self) -> f32 {
        self.chip_rate * self.samples_per_chip as f32
    }

    pub fn validate(&self) {
        assert!(self.spreading_factor > 0, "spreading_factor must be > 0");
        assert!(self.chip_rate > 0.0, "chip_rate must be > 0");
        assert!(self.samples_per_chip > 0, "samples_per_chip must be > 0");
        assert!(!self.prn_code.is_empty(), "prn_code must not be empty");
    }
}

/// Simple DSSS encoder.
pub struct DsssEncoder {
    config: DsssConfig,
}

impl DsssEncoder {
    pub fn new(config: DsssConfig) -> Self {
        config.validate();
        Self { config }
    }

    /// Encode raw bytes into DSSS chips in baseband (one float per chip).
    pub fn encode_to_chips(&self, data: &[u8]) -> Vec<f32> {
        let bits = bytes_to_bits(data);
        let mut chips = Vec::with_capacity(bits.len() * self.config.spreading_factor);

        for bit in bits {
            let bit_val = if bit == 1 { 1.0_f32 } else { -1.0_f32 };

            for j in 0..self.config.spreading_factor {
                let code_val = self.config.prn_code[j % self.config.prn_code.len()] as f32;
                let chip = bit_val * code_val;
                chips.push(chip);
            }
        }

        chips
    }

    /// Encode raw bytes and produce a real passband signal centered at carrier_freq.
    ///
    /// Output: Vec<f32> of samples at sample_rate = chip_rate * samples_per_chip.
    pub fn encode_to_passband(&self, data: &[u8]) -> Vec<f32> {
        let chips = self.encode_to_chips(data);
        let sample_rate = self.config.sample_rate();
        let dt = 1.0_f32 / sample_rate;

        let total_samples = chips.len() * self.config.samples_per_chip;
        let mut samples = Vec::with_capacity(total_samples);

        let mut t = 0.0_f32;
        let omega_c = 2.0_f32 * PI * self.config.carrier_freq;

        for chip in chips {
            for _ in 0..self.config.samples_per_chip {
                let carrier = (omega_c * t).cos();
                let s = chip * carrier;
                samples.push(s);
                t += dt;
            }
        }

        samples
    }
}

/// Simple DSSS decoder (assumes perfect timing and carrier).
pub struct DsssDecoder {
    config: DsssConfig,
}

impl DsssDecoder {
    pub fn new(config: DsssConfig) -> Self {
        config.validate();
        Self { config }
    }

    /// Decode from DSSS chips (baseband) to raw bytes.
    ///
    /// Input: one float per chip, length must be multiple of spreading_factor.
    pub fn decode_from_chips(&self, chips: &[f32]) -> Vec<u8> {
        let n = self.config.spreading_factor;
        assert!(
            chips.len() % n == 0,
            "chip stream length must be multiple of spreading_factor"
        );

        let mut bits = Vec::with_capacity(chips.len() / n);

        for symbol_chips in chips.chunks(n) {
            let mut acc = 0.0_f32;
            for (j, &chip) in symbol_chips.iter().enumerate() {
                let code_val = self.config.prn_code[j % self.config.prn_code.len()] as f32;
                acc += chip * code_val;
            }
            // Matched-filter decision: sign(acc) -> bit
            let bit = if acc >= 0.0 { 1_u8 } else { 0_u8 };
            bits.push(bit);
        }

        bits_to_bytes(&bits)
    }

    /// Decode from a real passband signal at the configured carrier frequency.
    ///
    /// Input: samples at sample_rate = chip_rate * samples_per_chip.
    /// Output: recovered bytes.
    pub fn decode_from_passband(&self, samples: &[f32]) -> Vec<u8> {
        let spc = self.config.samples_per_chip;
        assert!(
            samples.len() % spc == 0,
            "sample length must be multiple of samples_per_chip"
        );

        let sample_rate = self.config.sample_rate();
        let dt = 1.0_f32 / sample_rate;
        let omega_c = 2.0_f32 * PI * self.config.carrier_freq;

        let mut chips = Vec::with_capacity(samples.len() / spc);

        // Coherent demod: multiply by cos(ωc t) and integrate over each chip.
        let mut t0 = 0.0_f32;
        for chip_samples in samples.chunks(spc) {
            let mut acc = 0.0_f32;

            for (i, &s) in chip_samples.iter().enumerate() {
                let t = t0 + (i as f32) * dt;
                let carrier = (omega_c * t).cos();
                acc += s * carrier;
            }

            t0 += (spc as f32) * dt;
            chips.push(acc);
        }

        self.decode_from_chips(&chips)
    }
}

/// Utility: convert bytes to bits (MSB-first).
fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &b in data {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    bits
}

/// Utility: convert bits to bytes (MSB-first). Any trailing bits < 8 are dropped.
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len() / 8);

    for chunk in bits.chunks(8) {
        if chunk.len() < 8 {
            break;
        }
        let mut b = 0_u8;
        for &bit in chunk {
            b = (b << 1) | (bit & 1);
        }
        bytes.push(b);
    }

    bytes
}

/// Helper: generate a ±1 PRN code from a seed.
pub fn generate_prn_from_seed(length: usize, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..length)
        .map(|_| if rng.gen_bool(0.5) { 1_i8 } else { -1_i8 })
        .collect()
}

fn main() {
    let spreading_factor = 4;
    let chip_rate = 200.0;
    let carrier_freq = 1000.0;
    let samples_per_chip = 4;

    // This is your user-selectable channel seed:
    let seed = b"My DSSS channel key";

    // Use ChaCha12 to produce a spreading code.
    let prn = generate_prn_chacha12(spreading_factor, seed);

    let config = DsssConfig {
        spreading_factor,
        chip_rate,
        carrier_freq,
        samples_per_chip,
        prn_code: prn,
    };

    let encoder = DsssEncoder::new(config.clone());
    let decoder = DsssDecoder::new(config);

    let message = b"Hello DSSS world!";
    let tx = encoder.encode_to_passband(message);
    let rx = decoder.decode_from_passband(&tx);

    println!("Recovered message: {}", String::from_utf8_lossy(&rx));
}

