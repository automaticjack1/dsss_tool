use clap::{ValueEnum, Parser};
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use image::{GrayImage, Luma};
use rand_chacha::ChaCha12Rng;
use rand::{Rng, SeedableRng};
use reed_solomon::{Encoder, Decoder};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{Read, Write};

// Near top of main.rs
const RS_DATA_LEN: usize = 64;   // data bytes per codeword
const RS_ECC_LEN: usize  = 191;    // parity bytes
const RS_CODEWORD_LEN: usize = RS_DATA_LEN + RS_ECC_LEN; // 255

// Example: fixed preamble bits (Barker-like or just pseudo-random)
const PREAMBLE_BITS: &[u8] = &[
    1,0,1,1,0,1,0,0,  1,1,0,0,1,0,1,1,
    0,1,0,1,1,0,0,1,  1,0,1,0,0,1,1,0,
];

type StereoSample = [f32; 2];

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Mode {
    Encode,
    Decode,
    Embed,
    DecodeWav,
}

#[derive(ValueEnum, Clone, Debug)]
enum Channel {
    Left,
    Right,
}

#[derive(Parser, Debug)]
#[command(name = "smtp-gated-relay", version, about)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,

    #[arg(long)]
    input: String,

    #[arg(long)]
    output: String,

    /// Payload file whose bytes will be embedded via DSSS
    #[arg(long)]
    payload: Option<String>,

    #[arg(long, env = "SEED", default_value = "My DSSS channel key")]
    seed: String,

    #[arg(long, default_value_t = 32)]
    spreading_factor: usize,

    /// Carrier frequency in Hz (must respect Nyquist for the input audio).
    #[arg(long, default_value_t = 16_000.0)]
    carrier_freq: f32,

    /// Chip rate in chips/s (used in encode/decode modes; embed mode derives chip_rate from audio Fs).
    #[arg(long, default_value_t = 10_000.0)]
    chip_rate: f32,

    /// Oversampling factor: samples per chip (integer).
    #[arg(long, default_value_t = 4)]
    samples_per_chip: usize,

    /// DSSS amplitude in dBFS (e.g. -20.0, -30.0).
    #[arg(long, default_value_t = -30.0, allow_negative_numbers = true)]
    dsss_dbfs: f32,

    /// Fraction of DSSS length to use as delay between L & R (0..1).
    #[arg(long, default_value_t = 0.5)]
    delay_fraction: f32,

    /// Plaintext message to encode in encode mode.
    #[arg(long, default_value = "Hello DSSS world!")]
    message: String,

    /// If set, generate spectrogram PNGs in embed mode.
    #[arg(long)]
    visualize: bool,

    /// Which channel to use when decoding from WAV (decode-wav mode).
    #[arg(long, value_enum, default_value_t = Channel::Left)]
    channel: Channel,
}

/// Choose the chip stream (cos, sin, -cos, -sin) with strongest correlation
/// against the known preamble.
fn select_best_chip_stream(chip_sets: &[Vec<f32>;4], preamble: &[f32]) -> Vec<f32> {
    let mut best_score = f32::NEG_INFINITY;
    let mut best: Option<&Vec<f32>> = None;

    for chips in chip_sets {
        if chips.len() < preamble.len() { continue; }

        let mut score = 0.0;
        for i in 0..preamble.len() {
            score += chips[i] * preamble[i];
        }

        if score.abs() > best_score {
            best_score = score.abs();
            best = Some(chips);
        }
    }

    best.unwrap().clone()
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
    }
}

fn design_bandpass_fir(
    sample_rate: f32,
    center_freq: f32,
    chip_rate: f32,
    taps: usize,
) -> Vec<f32> {
    let nyquist = sample_rate * 0.5;

    // Narrower bandwidth: ~1.0 * chip_rate (trades a bit of DSSS energy for better SNR)
    let mut half_bw = 0.5 * chip_rate; // you can experiment: 0.5, 0.75, 1.0
    if half_bw > 0.9 * nyquist {
        half_bw = 0.9 * nyquist;
    }

    let f1 = (center_freq - half_bw).max(0.0);
    let f2 = (center_freq + half_bw).min(nyquist);

    let fc1 = f1 / sample_rate; // normalized (0..0.5)
    let fc2 = f2 / sample_rate;

    let m = (taps - 1) as f32 / 2.0;
    let mut h = vec![0.0f32; taps];

    for n in 0..taps {
        let k = n as f32 - m;
        // Ideal bandpass = (lowpass at fc2) - (lowpass at fc1)
        let lp2 = 2.0 * fc2 * sinc(2.0 * fc2 * k);
        let lp1 = 2.0 * fc1 * sinc(2.0 * fc1 * k);
        let ideal = lp2 - lp1;

        // Hamming window
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (taps - 1) as f32).cos();

        h[n] = ideal * w;
    }

    // Normalize to unit L1 gain; absolute scale isn’t critical since demod is relative,
    // but this keeps values reasonable.
    let sum: f32 = h.iter().sum();
    if sum.abs() > 1e-8 {
        for v in &mut h {
            *v /= sum;
        }
    }

    h
}

fn apply_fir(input: &[f32], h: &[f32]) -> Vec<f32> {
    let n = input.len();
    let m = h.len();
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let mut acc = 0.0;
        let mut k = 0usize;
        while k < m && k <= i {
            acc += input[i - k] * h[k];
            k += 1;
        }
        out[i] = acc;
    }

    out
}

fn bits_to_chips(bits: &[u8], cfg: &DsssConfig) -> Vec<f32> {
    let sf = cfg.spreading_factor;
    let prn = &cfg.prn_code;
    let mut chips = Vec::with_capacity(bits.len() * sf);

    for &b in bits {
        let symbol = if b == 0 { -1.0 } else { 1.0 };
        for j in 0..sf {
            let pn = prn[j % prn.len()] as f32;
            chips.push(symbol * pn);
        }
    }

    chips
}

fn chips_to_passband(chips: &[f32], cfg: &DsssConfig) -> Vec<f32> {
    let sample_rate = cfg.sample_rate();
    let dt = 1.0_f32 / sample_rate;

    let total_samples = chips.len() * cfg.samples_per_chip;
    let mut samples = Vec::with_capacity(total_samples);

    let mut t = 0.0_f32;
    let omega_c = 2.0_f32 * PI * cfg.carrier_freq;

    for &chip in chips {
        for _ in 0..cfg.samples_per_chip {
            let carrier = (omega_c * t).cos();
            let s = chip * carrier;
            samples.push(s);
            t += dt;
        }
    }

    samples
}

/// Build the known preamble chips sequence for acquisition.
fn preamble_chips(cfg: &DsssConfig) -> Vec<f32> {
    bits_to_chips(PREAMBLE_BITS, cfg)
}

/// Find chip offset of preamble in `chips` using sliding correlation.
/// Returns Some(offset) if found, else None.
fn acquire_preamble_offset(
    pre_chips: &[f32],
    chips: &[f32],
    max_search: usize,
) -> Option<usize> {
    let n = pre_chips.len();
    if n == 0 || chips.len() < n {
        return None;
    }

    // We’ll search up to this many starting positions.
    //let max_pos = chips.len().saturating_sub(n).min(max_search);
    let max_pos = max_search.min(chips.len().saturating_sub(n));

    // Energy of the preamble (for ±1 chips this is just n, but we’ll compute it)
    let pre_energy: f32 = pre_chips.iter().map(|x| x * x).sum();

    let mut best_idx: usize = 0;
    let mut best_score: f32 = f32::NEG_INFINITY;

    for start in 0..=max_pos {
        let mut acc = 0.0f32;
        // Correlate pre_chips with chips[start .. start+n]
        for i in 0..n {
            acc += pre_chips[i] * chips[start + i];
        }
        if acc > best_score {
            best_score = acc;
            best_idx = start;
        }
    }

    // Require a strong match; anything below this is treated as "no preamble".
    // For a clean match, best_score ≈ pre_energy (~n). Random junk is << pre_energy.
    //let threshold = 0.7 * pre_energy;
    let threshold = 0.25 * pre_energy;

    if best_score < threshold {
        eprintln!(
            "[decode] no strong preamble found: best_score={} threshold={} (energy={})",
            best_score, threshold, pre_energy
        );
        None
    } else {
        println!(
            "[decode] preamble peak at chip offset {} with score {} (energy={})",
            best_idx, best_score, pre_energy
        );
        Some(best_idx)
    }
}

/// Compute a spectrogram (waterfall) of mono samples and save to a PNG.
/// - `samples`: mono PCM in [-1,1]
/// - `sample_rate`: sample rate in Hz
/// - `n_fft`: FFT size
/// - `hop`: hop size (overlap = n_fft - hop)
fn save_spectrogram_png(
    samples: &[f32],
    _sample_rate: u32,
    n_fft: usize,
    hop: usize,
    path: &str,
) {
    assert!(n_fft.is_power_of_two(), "n_fft must be power of 2");
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Compute number of frames
    let num_frames = if samples.len() < n_fft {
        0
    } else {
        1 + (samples.len() - n_fft) / hop
    };

    if num_frames == 0 {
        eprintln!("Not enough samples for spectrogram");
        return;
    }

    let mut spectrogram: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

    let mut window = vec![0.0_f32; n_fft];
    let mut spectrum = vec![Complex32::new(0.0, 0.0); n_fft];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop;
        let end = start + n_fft;
        if end > samples.len() {
            break;
        }

        // Hann window
        for i in 0..n_fft {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / (n_fft as f32)).cos();
            window[i] = samples[start + i] * w;
            spectrum[i] = Complex32::new(window[i], 0.0);
        }

        fft.process(&mut spectrum);

        // Magnitude
        let mut mags = Vec::with_capacity(n_fft / 2);
        for i in 0..(n_fft / 2) {
            let m = spectrum[i].norm();
            mags.push(m);
        }

        spectrogram.push(mags);
    }

    // Convert to image using log-magnitude
    let height = n_fft as u32 / 2;
    let width = spectrogram.len() as u32;
    let mut img = GrayImage::new(width, height);

    // Find global min/max of log magnitudes
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for row in &spectrogram {
        for &m in row {
            let v = (m + 1e-12).ln(); // log
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
    }

    // Render: time → x, frequency → y (0 = DC at bottom)
    for (x, row) in spectrogram.iter().enumerate() {
        for (y, &m) in row.iter().enumerate() {
            let v = (m + 1e-12).ln();
            let norm = if max_val > min_val {
                (v - min_val) / (max_val - min_val)
            } else {
                0.0
            };
            let pixel_val = (norm.clamp(0.0, 1.0) * 255.0) as u8;
            let py = (height - 1 - y as u32).min(height - 1);
            img.put_pixel(x as u32, py, Luma([pixel_val]));
        }
    }

    if let Err(e) = img.save(path) {
        eprintln!("Failed to save spectrogram to {}: {}", path, e);
    } else {
        println!("Spectrogram written to {}", path);
    }
}

/// Read a stereo WAV file into Vec<[f32; 2]> and return (samples, sample_rate).
fn read_stereo_wav(path: &str) -> (Vec<StereoSample>, u32) {
    let mut reader = WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();

    assert_eq!(spec.channels, 2, "Input WAV must be stereo");

    let sr = spec.sample_rate;
    let format = spec.sample_format;
    let bits = spec.bits_per_sample;

    println!(
        "[embed] Input WAV: {} Hz, {}-bit {:?}",
        sr, bits, format
    );

    let samples: Vec<StereoSample> = match (format, bits) {
        // 16-bit signed integer PCM
        (SampleFormat::Int, 16) => {
            reader
                .samples::<i16>()
                .collect::<Result<Vec<i16>, _>>()
                .expect("Failed to read 16-bit samples")
                .chunks_exact(2)
                .map(|ch| {
                    let l = ch[0] as f32 / i16::MAX as f32;
                    let r = ch[1] as f32 / i16::MAX as f32;
                    [l, r]
                })
                .collect()
        }

        // 24-bit signed integer PCM (stored in 32-bit container)
        (SampleFormat::Int, 24) => {
            // 24-bit max value: 2^(24-1)-1 = 8_388_607
            const MAX_24: f32 = 8_388_607.0;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .expect("Failed to read 24-bit samples (as i32)")
                .chunks_exact(2)
                .map(|ch| {
                    let l = ch[0] as f32 / MAX_24;
                    let r = ch[1] as f32 / MAX_24;
                    [l, r]
                })
                .collect()
        }

        // 32-bit signed integer PCM
        (SampleFormat::Int, 32) => {
            reader
                .samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .expect("Failed to read 32-bit samples")
                .chunks_exact(2)
                .map(|ch| {
                    let l = ch[0] as f32 / i32::MAX as f32;
                    let r = ch[1] as f32 / i32::MAX as f32;
                    [l, r]
                })
                .collect()
        }

        // 32-bit float PCM (already -1.0..1.0 typically)
        (SampleFormat::Float, 32) => {
            reader
                .samples::<f32>()
                .collect::<Result<Vec<f32>, _>>()
                .expect("Failed to read 32-bit float samples")
                .chunks_exact(2)
                .map(|ch| {
                    let l = ch[0];
                    let r = ch[1];
                    [l, r]
                })
                .collect()
        }

        // You can add more formats if you really want
        _ => {
            panic!(
                "Unsupported WAV format: {:?} with {} bits per sample",
                format, bits
            );
        }
    };

    (samples, sr)
}

/// Write a stereo WAV file from Vec<[f32; 2]>.
fn write_stereo_wav(path: &str, samples: &[StereoSample], sample_rate: u32) {
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV file");

    for &[l, r] in samples {
        let il = (l.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let ir = (r.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(il).expect("Failed to write sample");
        writer.write_sample(ir).expect("Failed to write sample");
    }

    writer.finalize().expect("Failed to finalize WAV file");
}

/// Mix a mono DSSS signal into stereo audio with a fixed delay between L and R.
/// The DSSS signal is scaled to a desired dBFS.
fn mix_dsss_stereo_delayed(
    audio: &[StereoSample],
    dsss: &[f32],
    delay_samples: usize,
    dsss_dbfs: f32,
) -> Vec<StereoSample> {
    // Convert dBFS to a linear scale factor based on DSSS peak
    let max_dsss = dsss
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| if v.abs() > m { v.abs() } else { m });

    let desired_peak = 10f32.powf(dsss_dbfs / 20.0);
    let scale = if max_dsss > 0.0 {
        desired_peak / max_dsss
    } else {
        0.0
    };

    println!(
        "[mix] max_dsss = {}, dsss_dbfs = {}, scale = {}",
        max_dsss, dsss_dbfs, scale
    );

    let mut out = Vec::with_capacity(audio.len());

    for (i, &[l0, r0]) in audio.iter().enumerate() {
        // Primary DSSS (no delay) for left channel
        let d0 = if i < dsss.len() {
            dsss[i] * scale
        } else {
            0.0
        };

        // Delayed DSSS for right channel
        let j = i.saturating_sub(delay_samples);
        let d1 = if j < dsss.len() {
            dsss[j] * scale
        } else {
            0.0
        };

        // NOTE: don't multiply by `scale` again – d0/d1 are already scaled
        let l = l0 + d0;
        let r = r0 + d1;

        out.push([l, r]);
    }

    out
}

/// Generate a ±1 PRN code using ChaCha12Rng.
/// The seed can be any byte slice; it will be zero-padded or truncated to 32 bytes (256 bits).
pub fn generate_prn_chacha12(length: usize, seed: &[u8]) -> Vec<i8> {
    let mut key = [0u8; 32];

    // Copy user seed into key (truncate or pad)
    for (i, &b) in seed.iter().take(32).enumerate() {
        key[i] = b;
    }

    let mut rng = ChaCha12Rng::from_seed(key);
    let mut prn = Vec::with_capacity(length);
    for _ in 0..length {
        let bit = rng.gen_bool(0.5); // true or false with equal probability
        prn.push(if bit { 1i8 } else { -1i8 });
    }
    prn
}

/// Spread-spectrum configuration for both encoder and decoder.
#[derive(Clone)]
pub struct DsssConfig {
    /// Spreading factor: chips per data bit.
    pub spreading_factor: usize,
    /// Chip rate (chips per second) in baseband.
    pub chip_rate: f32,
    /// Carrier frequency (Hz).
    pub carrier_freq: f32,
    /// Oversampling factor: number of samples per chip in passband.
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
        chips_to_passband(&chips, &self.config)
    }
}

/// Simple DSSS decoder (assumes perfect carrier).
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

	let prn = &self.config.prn_code;
	let n_bits = chips.len() / n;
	let mut bits = Vec::with_capacity(n_bits);

	for bit_idx in 0..n_bits {
            let mut acc = 0.0;
            for k in 0..n {
		let chip = chips[bit_idx * n + k];
		let pn   = prn[k % prn.len()] as f32; // +1 / -1
		acc += chip * pn;
            }
            bits.push(if acc >= 0.0 { 1u8 } else { 0u8 });
	}
	
	Self::bits_to_bytes(&bits)
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
	let mut out = Vec::with_capacity((bits.len() + 7) / 8);
	let mut byte = 0u8;
	for (i, &b) in bits.iter().enumerate() {
            byte = (byte << 1) | (b & 1);
            if i % 8 == 7 {
		out.push(byte);
		byte = 0;
            }
	}
	let rem = bits.len() % 8;
	if rem != 0 {
            byte <<= 8 - rem;
            out.push(byte);
	}
	out
    }

    /// Demodulate passband samples back down to DSSS chips (baseband).
    /// Fully noncoherent demodulation.
    /// Returns a Vec of 4 chip sequences:
    /// [chips_cos, chips_sin, chips_neg_cos, chips_neg_sin]
    pub fn demodulate_to_chips_nco(&self, samples: &[f32]) -> [Vec<f32>; 4] {
	let spc = self.config.samples_per_chip;
	let sample_rate = self.config.sample_rate();
	let dt = 1.0_f32 / sample_rate;
	let omega_c = 2.0_f32 * PI * self.config.carrier_freq;

	let mut chips_cos      = Vec::with_capacity(samples.len() / spc.max(1));
	let mut chips_sin      = Vec::with_capacity(samples.len() / spc.max(1));
	let mut chips_negcos   = Vec::with_capacity(samples.len() / spc.max(1));
	let mut chips_negsin   = Vec::with_capacity(samples.len() / spc.max(1));

	let mut t0 = 0.0_f32;

	for chip_samples in samples.chunks(spc) {
            if chip_samples.len() < spc { break; }

            let mut acc_cos = 0.0_f32;
            let mut acc_sin = 0.0_f32;

            for (i, &s) in chip_samples.iter().enumerate() {
		let t = t0 + (i as f32) * dt;
		let c =  (omega_c * t).cos();
		let q =  (omega_c * t).sin();
		acc_cos += s * c;
		acc_sin += s * q;
            }

            chips_cos.push(     acc_cos);
            chips_sin.push(     acc_sin);
            chips_negcos.push( -acc_cos);
            chips_negsin.push( -acc_sin);

            t0 += (spc as f32) * dt;
	}

	[chips_cos, chips_sin, chips_negcos, chips_negsin]
    }

    /// Decode from a real passband signal assuming chip timing is
    /// already aligned.
    /// Fully noncoherent: select best of the four demod streams.
    pub fn decode_from_passband_no_preamble(&self, samples: &[f32]) -> Vec<u8> {
	// 1. Demodulate into 4 chip streams
	let chip_sets = self.demodulate_to_chips_nco(samples);

	// 2. Build preamble chips for matching
	let pre = preamble_chips(&self.config);

	// 3. Select the strongest of the four streams
	let chips = select_best_chip_stream(&chip_sets, &pre);

	// ------- Now 'chips' is Vec<f32> — just like before ---------

	let chips_per_bit = self.config.spreading_factor;
	let codeword_bits = RS_CODEWORD_LEN * 8;

	let needed_chips = (codeword_bits + PREAMBLE_BITS.len()) * chips_per_bit;

	if chips.len() < needed_chips {
            println!(
		"[debug] not enough chips: have {}, need {}",
		chips.len(),
		needed_chips
            );
	}

	// Assume frame starts at chip 0
	let start = PREAMBLE_BITS.len() * chips_per_bit;
	let end = start + codeword_bits * chips_per_bit;

	if end > chips.len() {
            println!("[debug] truncated chips: requested {}..{}", start, end);
	}

	let end_clamped = end.min(chips.len());

	let payload_chips = &chips[start..end_clamped];

	self.decode_from_chips(payload_chips)
    }

    /// Decode from passband using a sliding-window correlator against the known
    /// preamble. This can acquire timing even if the stream starts at an
    /// arbitrary position, e.g., after clipping.
    pub fn decode_from_passband_with_preamble(&self, samples: &[f32], taps:usize) -> Option<Vec<u8>> {
	// 1. Passband -> chips
	let chip_sets = self.demodulate_to_chips_nco(samples);
	let pre = preamble_chips(&self.config);
	let chips = select_best_chip_stream(&chip_sets, &pre);

	// 2. Known preamble
	let pre_chips = preamble_chips(&self.config);
	if chips.len() < pre_chips.len() {
            eprintln!("[decode] not enough chips for preamble");
            return None;
	}

	// 3. Sliding-window acquisition
	    //
	// In EMBED mode we always start the frame at t=0 in the audio, so the
	// preamble should be within a small window after the start (plus FIR
	// group delay). To avoid locking on spurious peaks later, restrict the
	// search region.
	let fir_group_delay_samples = taps / 2; // == 128 for taps = 257
	let fir_group_delay_chips = fir_group_delay_samples / self.config.samples_per_chip;

	// e.g. search up to 4 preamble-lengths beyond the expected position
	let small_search = pre_chips.len() * 4 + fir_group_delay_chips;
	//let max_search = chips.len().saturating_sub(pre_chips.len());

	let offset = acquire_preamble_offset(&pre_chips, &chips, small_search)?;
	println!(
            "[decode] acquired preamble at chip offset {} (of {})",
            offset,
            chips.len()
	);

	// 4. Only decode one RS codeword: RS_CODEWORD_LEN bytes → RS_CODEWORD_LEN * 8 bits
	let chips_per_bit = self.config.spreading_factor;

	// raw chip index just after the preamble
	let start = offset + pre_chips.len();

	println!(
	    "[decode] len(chips) = {}, offset = {}, pre_chips.len() = {}, start = {}",
	    chips.len(),
	    offset,
	    pre_chips.len(),
	    start
	);
	
	if start >= chips.len() {
            eprintln!("[decode] no chips after preamble");
            return None;
	}

	let codeword_bits = RS_CODEWORD_LEN * 8;
	let min_needed_chips = start + codeword_bits * chips_per_bit;

	if chips.len() < min_needed_chips {
            eprintln!(
		"[decode] warning: only {} chips after preamble, need {} for full RS codeword",
		chips.len() - start,
		codeword_bits * chips_per_bit
            );
	}

	let bits_available = (chips.len() - start) / chips_per_bit;
	let bits_to_decode = bits_available.min(codeword_bits);

	if bits_to_decode < codeword_bits {
            eprintln!(
		"[decode] warning: only {} bits available, expected {}; frame truncated",
		bits_to_decode, codeword_bits
            );
	}

	let chips_to_decode = bits_to_decode * chips_per_bit;
	let payload_chips = &chips[start..start + chips_to_decode];

	// 5. Despread to bits/bytes
	let bytes = self.decode_from_chips(payload_chips);
	if bytes.len() < RS_CODEWORD_LEN {
            eprintln!(
		"[decode] only {} bytes recovered, expected {}; aborting",
		bytes.len(),
		RS_CODEWORD_LEN
            );
            return None;
	}

	println!(
	    "[debug] pre-RS first 32 bytes: {:02X?}",
	    &bytes[..32.min(bytes.len())]
	);

	// TEMP: compare against known transmitted codeword if available
	if let Ok(tx_cw) = std::fs::read("tx_codeword.bin") {
	    if tx_cw.len() == RS_CODEWORD_LEN {
		let mut byte_errors = 0usize;
		let mut bit_errors  = 0usize;

		for i in 0..RS_CODEWORD_LEN {
		    if tx_cw[i] != bytes[i] {
			byte_errors += 1;
			bit_errors += (tx_cw[i] ^ bytes[i]).count_ones() as usize;
		    }
		}

		println!(
		    "[debug] codeword mismatch: {} / {} bytes wrong, ~{} bits wrong",
		    byte_errors,
		    RS_CODEWORD_LEN,
		    bit_errors
		);
	    } else {
		println!(
		    "[debug] tx_codeword.bin len {} != RS_CODEWORD_LEN {}",
		    tx_cw.len(),
		    RS_CODEWORD_LEN
		);
	    }
	}

	// Take first RS_CODEWORD_LEN bytes as RS codeword
	let mut codeword = [0u8; RS_CODEWORD_LEN];
	codeword.copy_from_slice(&bytes[..RS_CODEWORD_LEN]);

	// 6. RS decode
	let dec = Decoder::new(RS_ECC_LEN);
	let mut buf = codeword;
	let recovered = match dec.correct(&mut buf, None) {
            Ok(r) => r,
            Err(e) => {
		eprintln!("[decode] RS decode failed: {:?}", e);
		return None;
            }
	};

	let data = recovered.data(); // &[u8], length RS_DATA_LEN
	if data.len() < 4 {
            eprintln!("[decode] RS data too short for length header");
            return None;
	}

	let payload_len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
	if payload_len > RS_DATA_LEN - 4 {
            eprintln!(
		"[decode] absurd payload length {} (max {})",
		payload_len,
		RS_DATA_LEN - 4
            );
            return None;
	}

	if data.len() < 4 + payload_len {
            eprintln!(
		"[decode] RS data too short for declared payload length {}",
		payload_len
            );
            return None;
	}

	let payload = &data[4..4 + payload_len];
	Some(payload.to_vec())
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

fn main() {
    let args = Args::parse();

    let taps = 257; // or 513 if you want sharper skirts
    let spreading_factor = args.spreading_factor;
    let chip_rate = args.chip_rate; // only used in encode/decode modes
    let carrier_freq = args.carrier_freq;
    let samples_per_chip = args.samples_per_chip;

    // User-selectable channel seed:
    let seed = args.seed.as_bytes();

    // Use ChaCha12 to produce a PRNG spreading code.
    let prn = generate_prn_chacha12(spreading_factor, seed);

    let config = DsssConfig {
        spreading_factor,
        chip_rate,
        carrier_freq,
        samples_per_chip,
        prn_code: prn,
    };

    let encoder = DsssEncoder::new(config.clone());
    let decoder = DsssDecoder::new(config.clone());

    //----------------------------------------------------------
    // MODE: ENCODE
    //----------------------------------------------------------
    if matches!(args.mode, Mode::Encode) {
        println!("[encode] reading plaintext message…");

        let msg_bytes = args.message.as_bytes();

        println!("[encode] encoding {} bytes…", msg_bytes.len());
        let tx_samples = encoder.encode_to_passband(msg_bytes);

        println!("[encode] writing {} float samples to {}", 
                 tx_samples.len(), args.output);

        // Write raw f32 samples (little-endian)
        let mut f = File::create(&args.output)
            .expect("failed to create output file");
        // SAFETY: f32 has no invalid bit patterns
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(
                tx_samples.as_ptr() as *const u8,
                tx_samples.len() * std::mem::size_of::<f32>(),
            )
        };

        f.write_all(raw_bytes).expect("failed to write samples");

        println!("[encode] done.");
        return;
    }

    //----------------------------------------------------------
    // MODE: DECODE
    //----------------------------------------------------------
    if matches!(args.mode, Mode::Decode) {
        println!("[decode] reading passband samples from {}", args.input);

        // Read the file into memory
        let mut raw = Vec::new();
        File::open(&args.input)
            .expect("failed to open input file")
            .read_to_end(&mut raw)
            .expect("failed to read samples");

        if raw.len() % 4 != 0 {
            panic!("Input file size is not a multiple of 4 bytes (f32 samples).");
        }

        let num_samples = raw.len() / 4;
        let mut samples = Vec::with_capacity(num_samples);
        for chunk in raw.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let s = f32::from_le_bytes(bytes);
            samples.push(s);
        }

        println!("[decode] {} samples loaded…", samples.len());

        // Simple aligned decode:
        let recovered = decoder.decode_from_passband_with_preamble(&samples, taps);

        // Write decoded bytes to output file
	match recovered {
	    Some(ref bytes) => {
		println!(
		    "[debug] recovered {} bytes, writing to {}",
		    bytes.len(),
		    args.output
		);

		let mut f = std::fs::File::create(&args.output)
		    .expect("failed to create output file");
		use std::io::Write;
		f.write_all(&bytes).expect("failed to write output message");

		println!(
		    "[debug] as UTF-8 (lossy): {}",
		    String::from_utf8_lossy(&bytes)
		);
	    }
	    None => {
		eprintln!("[debug] loopback failed: no valid frame recovered");
	    }
	}

        return;
    }

    //----------------------------------------------------------
    // MODE: EMBED  (stereo audio, auto chip_rate, repeated frames)
    //----------------------------------------------------------
    if matches!(args.mode, Mode::Embed) {
        //--------------------------------------------------
        // 1. Read stereo audio & determine sample rate
        //--------------------------------------------------
        let (audio, wav_fs) = read_stereo_wav(&args.input);
        let wav_fs_f = wav_fs as f32;

        //--------------------------------------------------
        // 2. Auto-deduce chip_rate from sample rate
        //--------------------------------------------------
        let spc = args.samples_per_chip.max(1) as f32;
        let chip_rate = wav_fs_f / spc;

        println!(
            "[embed] Input Fs = {} Hz, auto chip_rate = {} chips/s, samples_per_chip = {}",
            wav_fs, chip_rate, args.samples_per_chip
        );

        //--------------------------------------------------
        // 3. Enforce Nyquist: carrier_freq < Fs/2
        //--------------------------------------------------
        let nyquist = wav_fs_f / 2.0;
        if args.carrier_freq <= 0.0 || args.carrier_freq >= nyquist {
            panic!(
                "carrier_freq={} Hz is invalid for Fs={} Hz (Nyquist = {} Hz). \
                 Choose 0 < carrier_freq < Nyquist.",
                args.carrier_freq, wav_fs, nyquist
            );
        }

        //--------------------------------------------------
        // 4. Build PRN & DSSS config
        //--------------------------------------------------
        let prn = generate_prn_chacha12(args.spreading_factor, args.seed.as_bytes());

        let config = DsssConfig {
            spreading_factor: args.spreading_factor,
            chip_rate,
            carrier_freq: args.carrier_freq,
            samples_per_chip: args.samples_per_chip,
            prn_code: prn,
        };

	let decoder = DsssDecoder::new(config.clone());

        //--------------------------------------------------
        // 5. Load payload bytes
        //--------------------------------------------------
        let payload_path = args
            .payload
            .as_ref()
            .expect("payload file is required in embed mode");
        let payload = std::fs::read(payload_path)
            .unwrap_or_else(|e| panic!("failed to read payload '{}': {e}", payload_path));

        println!(
            "[embed] Payload '{}' has {} bytes",
            payload_path,
            payload.len()
        );

        //--------------------------------------------------
        // 6. Build one DSSS frame (preamble + payload bits)
        //--------------------------------------------------
        let payload_len: usize = payload.len();

	// 6.1. Build [len|payload] as RS data
	if payload_len + 4 > RS_DATA_LEN {
	    eprintln!(
		"[embed] payload too long: {} bytes (max {} bytes, excluding length header)",
		payload_len,
		RS_DATA_LEN - 4
	    );
	    std::process::exit(1);
	}

	let mut rs_data = vec![0u8; RS_DATA_LEN];

	let len_u32: u32 = payload_len as u32;
	rs_data[0..4].copy_from_slice(&len_u32.to_be_bytes());
	rs_data[4..4 + payload_len].copy_from_slice(&payload);

	// 6.2. RS-encode: rs_data -> rs_codeword (len = RS_CODEWORD_LEN)
	let enc = Encoder::new(RS_ECC_LEN);
	let encoded = enc.encode(&rs_data[..]); // Buffer
	let rs_codeword: &[u8] = encoded.as_ref(); // [data|ecc], length 255

	// DEBUG: dump the exact codeword we transmitted
	std::fs::write("tx_codeword.bin", rs_codeword)
	    .expect("failed to write tx_codeword.bin");
	
	// RS SELF-TEST (no DSSS at all)
	{
	    println!("[debug] RS self-test: data.len() = {}, codeword.len() = {}",
		     rs_data.len(), rs_codeword.len());

	    let mut cw = rs_codeword.to_vec();
	    let dec = Decoder::new(RS_ECC_LEN);

	    match dec.correct(&mut cw, None) {
		Ok(recovered) => {
		    let data_back = recovered.data();
		    println!("[debug] RS self-test: recovered.data().len() = {}", data_back.len());

		    if data_back == &rs_data[..] {
			println!("[debug] RS self-test: SUCCESS (data matches)");
		    } else {
			println!("[debug] RS self-test: MISMATCH!");
			println!("[debug] orig[0..32]:   {:02X?}", &rs_data[..32.min(rs_data.len())]);
			println!("[debug] recov[0..32]: {:02X?}", &data_back[..32.min(data_back.len())]);
		    }
		}
		Err(e) => {
		    println!("[debug] RS self-test: FAILED with {:?}", e);
		}
	    }
	}
	
	// 6.3. Build DSSS frame bits: PREAMBLE_BITS + bits(rs_codeword)
	let mut frame_bits = Vec::with_capacity(PREAMBLE_BITS.len() + RS_CODEWORD_LEN * 8);
	frame_bits.extend_from_slice(PREAMBLE_BITS);
	frame_bits.extend(bytes_to_bits(rs_codeword));

	// 6.4. As before: bits -> chips -> passband
	let frame_chips = bits_to_chips(&frame_bits, &config);

	// DEBUG 1: chips-only loopback (bypass carrier/audio entirely)
	if false {
	    println!("[debug] running DSSS chips-only self-test…");

	    // preamble chips using the *same* config
	    let pre_chips = preamble_chips(&config);

	    println!(
		"[debug] frame_chips.len()={}, pre_chips.len()={}",
		frame_chips.len(),
		pre_chips.len()
	    );

	    // Sanity: does the frame actually *start* with the preamble chips?
	    let mut mismatch_at: Option<usize> = None;
	    for i in 0..pre_chips.len() {
		if (frame_chips[i] - pre_chips[i]).abs() > 1e-6 {
		    mismatch_at = Some(i);
		    break;
		}
	    }

	    match mismatch_at {
		Some(i) => {
		    println!(
			"[debug] preamble mismatch at chip {}: frame={} pre={}",
			i, frame_chips[i], pre_chips[i]
		    );
		}
		None => {
		    println!("[debug] preamble chips match at the start of frame_chips");
		}
	    }

	    // Now decode *just the payload chips* directly from chips
	    let chips_per_bit = config.spreading_factor;
	    let codeword_bits = RS_CODEWORD_LEN * 8;
	    let payload_start = pre_chips.len();
	    let payload_end = payload_start + codeword_bits * chips_per_bit;

	    if payload_end > frame_chips.len() {
		println!(
		    "[debug] not enough chips in frame for full codeword: have {}, need {}",
		    frame_chips.len(),
		    payload_end
		);
	    } else {
		let payload_chips = &frame_chips[payload_start..payload_end];

		let raw_bytes = decoder.decode_from_chips(payload_chips);
		println!(
		    "[debug] chips-only first 32 bytes: {:02X?}",
		    &raw_bytes[..32.min(raw_bytes.len())]
		);
	    }

	    std::process::exit(0);
	}

	// DEBUG 2: chips-only DSSS+RS loopback (no carrier, no audio)
	if false {
	    println!("[debug] running chips-only DSSS+RS self-test…");

	    let pre_chips = preamble_chips(&config);

	    println!(
		"[debug] frame_chips.len() = {}, pre_chips.len() = {}",
		frame_chips.len(),
		pre_chips.len()
	    );

	    // payload chips right after preamble
	    let chips_per_bit = config.spreading_factor;
	    let codeword_bits = RS_CODEWORD_LEN * 8;
	    let payload_start = pre_chips.len();
	    let payload_end   = payload_start + codeword_bits * chips_per_bit;

	    if payload_end > frame_chips.len() {
		println!(
		    "[debug] not enough chips in frame for full codeword: have {}, need {}",
		    frame_chips.len(),
		    payload_end
		);
	    } else {
		let payload_chips = &frame_chips[payload_start..payload_end];

		// Direct despread → bytes
		let raw_bytes = decoder.decode_from_chips(payload_chips);
		println!(
		    "[debug] chips-only first 32 bytes: {:02X?}",
		    &raw_bytes[..32.min(raw_bytes.len())]
		);

		// RS decode on chips-only result
		if raw_bytes.len() >= RS_CODEWORD_LEN {
		    let mut cw = [0u8; RS_CODEWORD_LEN];
		    cw.copy_from_slice(&raw_bytes[..RS_CODEWORD_LEN]);

		    let dec = Decoder::new(RS_ECC_LEN);
		    match dec.correct(&mut cw, None) {
			Ok(recovered) => {
			    let data = recovered.data();
			    println!(
				"[debug] chips-only RS decode data[0..32]: {:02X?}",
				&data[..32.min(data.len())]
			    );
			}
			Err(e) => {
			    println!("[debug] chips-only RS decode failed: {:?}", e);
			}
		    }
		} else {
		    println!(
			"[debug] chips-only raw_bytes too short: {} < {}",
			raw_bytes.len(),
			RS_CODEWORD_LEN
		    );
		}
	    }

	    std::process::exit(0);
	}

	let frame = chips_to_passband(&frame_chips, &config);
	println!("[debug] passband frame has {} samples", frame.len());

	// -------------------------------------------------------------
	// DEBUG 3: full float loopback: frame -> decode_with_preamble
	// -------------------------------------------------------------
	if true {
	    println!("[debug] running full-float DSSS+RS loopback (no WAV, no quantization)…");

	    match decoder.decode_from_passband_with_preamble(&frame, taps) {
		Some(bytes) => {
		    println!(
			"[debug] full-float loopback recovered {} bytes: {:?}",
			bytes.len(),
			String::from_utf8_lossy(&bytes)
		    );
		}
		None => {
		    println!("[debug] full-float loopback: no valid frame recovered");
		}
	    }

	    std::process::exit(0);
	}

	let rt_chips = decoder.demodulate_to_chips_nco(&frame);
	println!(
	    "[debug] passband roundtrip: frame_chips.len()={}, rt_chips[0].len()={}",
	    frame_chips.len(),
	    rt_chips[0].len()
	);

	// Compare first few chips numerically
	let n_show = frame_chips.len().min(rt_chips.len()).min(16);
	for i in 0..n_show {
	    println!(
		"[debug] chip[{}]: orig={:.4} rt={:.4}",
		i, frame_chips[i], rt_chips[0][i]
	    );
	}

	// Compute normalized correlation between original and round-tripped chips
	let mut num = 0.0f32;
	let mut den1 = 0.0f32;
	let mut den2 = 0.0f32;
	for i in 0..n_show {
	    let a = frame_chips[i];
	    let b = rt_chips[0][i];
	    num += a * b;
	    den1 += a * a;
	    den2 += b * b;
	}

	let corr = num / ((den1.sqrt() * den2.sqrt()).max(1e-9));
	println!("[debug] partial chip correlation (first {}): {}", n_show, corr);
	
	// DEBUG: loopback test of DSSS+RS without audio channel
	if false {
	    println!("[debug] running DSSS loopback self-test…");

	    // "Transmit" frame directly
	    let tx = frame.clone();

	    // "Receive" and decode directly
	    let recovered = decoder.decode_from_passband_with_preamble(&tx, taps);
	    let raw_bytes = decoder.decode_from_passband_no_preamble(&tx);
	    println!("[debug] first 32 bytes (no preamble): {:02X?}", &raw_bytes[..32.min(raw_bytes.len())]);
	    
	    match recovered {
		Some(bytes) => {
		    println!(
			"[debug] loopback recovered {} bytes: {:?}",
			bytes.len(),
			String::from_utf8_lossy(&bytes)
		    );
		}
		None => {
		    println!("[debug] loopback failed: no valid frame recovered");
		}
	    }

	    std::process::exit(0);
	}

        let audio_len = audio.len();
        let frame_len = frame.len();

        let n_frames = if frame_len == 0 {
            0
        } else {
            audio_len / frame_len
        };

        if n_frames == 0 {
            println!(
                "[embed] Warning: audio too short for a single DSSS frame ({} samples, need {}). No DSSS will be embedded.",
                audio_len, frame_len
            );
	    std::process::exit(1);
        }

        // Build DSSS vector matching audio length, tiling as many whole frames as fit.
        let mut dsss = vec![0.0_f32; audio_len];
        for f in 0..n_frames {
            let start = f * frame_len;
            let end = start + frame_len;
            if end > audio_len {
                break;
            }
            dsss[start..end].copy_from_slice(&frame[..(end - start)]);
        }

        println!(
            "[embed] DSSS frame length = {} samples, frames embedded = {}, total DSSS len = {}",
            frame_len,
            n_frames,
            dsss.len()
        );

        //--------------------------------------------------
        // 7. Compute delay between L & R in samples
        //--------------------------------------------------
        let delay_fraction = args.delay_fraction.clamp(0.0, 1.0);
        let delay_samples =
            ((frame_len as f32) * delay_fraction).round() as usize;

        println!(
            "[embed] Using delay_fraction = {:.3} → delay_samples = {}",
            delay_fraction, delay_samples
        );

        //--------------------------------------------------
        // 8. Mix DSSS into stereo audio
        //--------------------------------------------------
        let mixed =
            mix_dsss_stereo_delayed(&audio, &dsss, delay_samples, args.dsss_dbfs);

        let original_left: Vec<f32> = audio.iter().map(|[l, _]| *l).collect();
        let mixed_left: Vec<f32>    = mixed.iter().map(|[l, _]| *l).collect();

        if args.visualize {
            save_spectrogram_png(
                &original_left,
                wav_fs,
                1024,              // n_fft
                512,               // hop
                "orig_left.png",
            );

            save_spectrogram_png(
                &mixed_left,
                wav_fs,
                1024,
                512,
                "mixed_left.png",
            );
        }

	let mut max_val = 0.0f32;
	for [l, r] in &mixed {
	    max_val = max_val.max(l.abs());
	    max_val = max_val.max(r.abs());
	}
	println!("[debug] max mixed amplitude = {}", max_val);

        //--------------------------------------------------
        // 9. Write stereo WAV with embedded DSSS
        //--------------------------------------------------
        write_stereo_wav(&args.output, &mixed, wav_fs);
        println!("[embed] Wrote embedded audio to '{}'", args.output);
        return;
    }

    //----------------------------------------------------------
    // MODE: DECODE-WAV  (decode DSSS from a stereo WAV file)
    //----------------------------------------------------------
    if matches!(args.mode, Mode::DecodeWav) {
        // 1. Read stereo WAV and derive sample rate
        let (audio, wav_fs) = read_stereo_wav(&args.input);
        let wav_fs_f = wav_fs as f32;

        // 2. Auto-chip-rate from Fs and samples_per_chip (same as EMBED)
        let spc = args.samples_per_chip.max(1) as f32;
        let chip_rate = wav_fs_f / spc;
    
        println!(
            "[decode-wav] Input Fs = {} Hz, auto chip_rate = {} chips/s, samples_per_chip = {}",
            wav_fs, chip_rate, args.samples_per_chip
        );
    
        // 3. Nyquist safety for carrier
        let nyquist = wav_fs_f / 2.0;
        if args.carrier_freq <= 0.0 || args.carrier_freq >= nyquist {
            panic!(
                "carrier_freq={} Hz is invalid for Fs={} Hz (Nyquist = {} Hz). \
                 Choose 0 < carrier_freq < Nyquist.",
                args.carrier_freq, wav_fs, nyquist
            );
        }
    
        // 4. Rebuild PRN & DSSS config to match EMBED
        let prn = generate_prn_chacha12(args.spreading_factor, args.seed.as_bytes());
    
        let config = DsssConfig {
            spreading_factor: args.spreading_factor,
            chip_rate,
            carrier_freq: args.carrier_freq,
            samples_per_chip: args.samples_per_chip,
            prn_code: prn,
        };
    
        let decoder = DsssDecoder::new(config.clone());
    
        // 5. Select one channel (NO folding to mono)
        let mono: Vec<f32> = match args.channel {
            Channel::Left => {
                println!("[decode-wav] using LEFT channel");
                audio.iter().map(|[l, _]| *l).collect()
            }
            Channel::Right => {
                println!("[decode-wav] using RIGHT channel");
                audio.iter().map(|[_, r]| *r).collect()
            }
        };
    
        println!(
            "[decode-wav] mono stream has {} samples",
            mono.len()
        );

	// Design bandpass filter based on carrier + chip rate
	let bp = design_bandpass_fir(
	    wav_fs as f32,
	    config.carrier_freq,
	    config.chip_rate, // make sure this is set consistently with EMBED
	    taps,
	);

	println!(
	    "[decode-wav] bandpass: fc={} Hz, chip_rate={} Hz, taps={}",
	    config.carrier_freq, config.chip_rate, taps
	);

	//let filtered = apply_fir(&mono, &bp);
	let filtered = mono;

	let chips = decoder.demodulate_to_chips_nco(&filtered);
	for i in 0..16 {
	    println!("[debug] chip[{}] = {}", i, chips[0][i]);
	}

        // 6. Run preamble-based sliding-window decode
	match decoder.decode_from_passband_with_preamble(&filtered, taps) {
	    Some(bytes) => {
		println!(
		    "[decode-wav] recovered {} bytes, writing to {}",
		    bytes.len(),
		    args.output
		);
		std::fs::write(&args.output, &bytes)
		    .expect("failed to write decoded payload");
	    }
	    None => {
		eprintln!("[decode-wav] failed: no valid frame recovered (see RS / preamble logs above)");
	    }
	}
    
        return;
    }
    
    unreachable!("mode must be encode, decode, or embed");
}

