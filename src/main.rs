use clap::{Parser, ValueEnum};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use image::{GrayImage, Luma};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use reed_solomon::{Decoder, Encoder};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{Read, Write};

// Near top of main.rs
const RS_DATA_LEN: usize = 64; // data bytes per codeword
const RS_ECC_LEN: usize = 191; // parity bytes
const RS_CODEWORD_LEN: usize = RS_DATA_LEN + RS_ECC_LEN; // 255

// Preamble now generated with the separate tool, preamble_generator,
// which will hunt through large spaces to produce workable preambles.
// In this case, 96 bits would require more out of the preamble detector,
// possibly even source-code modifications to increase its confidence.
// 64 bits is sufficient.
const PREAMBLE_BITS: &[u8] = &[
    0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
    1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
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
#[command(name = "dsss_tool", version, about)]
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

    // Self Test mode
    #[arg(long)]
    selftest: bool,
}

/// Number of payload bits carried by one RS codeword
#[inline]
fn _payload_bits_len() -> usize {
    RS_CODEWORD_LEN * 8
}

fn ber_bytes(a: &[u8], b: &[u8]) -> (usize, usize, usize) {
    let n = a.len().min(b.len());
    let mut byte_err = 0usize;
    let mut bit_err = 0usize;
    for i in 0..n {
        let x = a[i] ^ b[i];
        if x != 0 {
            byte_err += 1;
        }
        bit_err += x.count_ones() as usize;
    }
    (n, byte_err, bit_err)
}

/// Total number of bits in one DSSS frame
/// (preamble + payload; extend here if you add CRC, frame IDs, etc.)
#[inline]
fn frame_bits_len() -> usize {
    PREAMBLE_BITS.len() + (RS_CODEWORD_LEN * 8)
}

/// Total number of chips in one DSSS frame
#[inline]
fn frame_period_chips(cfg: &DsssConfig) -> usize {
    frame_bits_len() * cfg.spreading_factor
}

/// Total number of passband samples in one DSSS frame
#[inline]
fn frame_period_samples(cfg: &DsssConfig) -> usize {
    frame_period_chips(cfg) * cfg.samples_per_chip
}

/// Choose the chip stream (cos, sin, -cos, -sin) with strongest correlation
/// against the known preamble.
fn select_best_chip_stream(chip_sets: &[Vec<f32>; 4], preamble: &[f32]) -> Vec<f32> {
    let pre_e: f32 = preamble.iter().map(|x| x * x).sum();
    if pre_e <= 1e-12 {
        return chip_sets[0].clone();
    }

    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for (idx, chips) in chip_sets.iter().enumerate() {
        if chips.len() < preamble.len() {
            continue;
        }

        // Score correlation over the first preamble-length window
        let mut score = 0.0f32;
        let mut win_e = 0.0f32;

        for i in 0..preamble.len() {
            let c = chips[i];
            let p = preamble[i];
            score += c * p;
            win_e += c * c;
        }

        let denom = (pre_e * win_e).sqrt().max(1e-12);
        let norm_score = score.abs() / denom;

        if norm_score > best_score {
            best_score = norm_score;
            best_idx = idx;
        }
    }

    chip_sets[best_idx].clone()
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
    }
}

fn design_lowpass_fir(sample_rate: f32, cutoff_hz: f32, taps: usize) -> Vec<f32> {
    // Odd taps → symmetric FIR → clean linear phase and integer sample group delay.
    // (You rely on gd=(taps-1)/2 later.)
    debug_assert!(
        taps % 2 == 1,
        "lowpass taps should be odd for symmetric FIR"
    );

    let nyquist = sample_rate * 0.5;
    let cutoff_hz = cutoff_hz.min(nyquist * 0.999);
    let fc = cutoff_hz / sample_rate;

    let m = (taps - 1) as f32 / 2.0;
    let mut h = vec![0.0f32; taps];

    for n in 0..taps {
        let k = n as f32 - m;
        // Ideal lowpass
        let ideal = 2.0 * fc * sinc(2.0 * fc * k);

        // Hamming window
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * n as f32 / (taps - 1) as f32).cos();

        h[n] = ideal * w;
    }

    // Normalize
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

    let total_samples = chips.len() * cfg.samples_per_chip;
    let mut samples = Vec::with_capacity(total_samples);

    let omega_dt = 2.0_f32 * PI * cfg.carrier_freq / sample_rate;
    let mut phase = 0.0f32;

    for &chip in chips {
        for _ in 0..cfg.samples_per_chip {
            let carrier = phase.cos();
            samples.push(chip * carrier);

            phase += omega_dt;
            if phase > PI {
                phase -= 2.0 * PI;
            } else if phase < -PI {
                phase += 2.0 * PI;
            }
        }
    }

    samples
}

/// Build the known preamble chips sequence for acquisition.
fn preamble_chips(cfg: &DsssConfig) -> Vec<f32> {
    bits_to_chips(PREAMBLE_BITS, cfg)
}

/// Find chip offset of preamble in `chips` using sliding correlation in the IQ
/// domain.
/// Returns a list of top candidate offsets and their normalized scores:
/// (offset, score, ci, cq). Highest score first. None if no candidates.
fn acquire_preamble_offset_iq(
    pre: &[f32],
    chips_i: &[f32],
    chips_q: &[f32],
) -> Option<Vec<(usize, f32, f32, f32)>> {
    let keep_n = 8usize;

    // Collect best candidate peaks
    let mut best: Vec<(usize, f32, f32, f32)> = Vec::new();
    // (idx, score, ci, cq)

    // Closure that maintains a top-N list by score
    let mut push_candidate = |idx: usize, score: f32, ci: f32, cq: f32| {
        best.push((idx, score, ci, cq));

        // Sort descending by score
        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Deduplicate identical offsets
        best.dedup_by(|a, b| a.0 == b.0);

        // Keep only top N
        if best.len() > keep_n {
            best.truncate(keep_n);
        }
    };

    // --- main scan loop ---
    let n = pre.len();
    let pre_energy: f32 = pre.iter().map(|x| x * x).sum();
    let max_pos = chips_i.len().saturating_sub(n);

    for start in 0..=max_pos {
        let mut ci = 0.0;
        let mut cq = 0.0;
        let mut win_e = 0.0;

        for k in 0..n {
            let i = chips_i[start + k];
            let q = chips_q[start + k];
            let p = pre[k];

            ci += p * i;
            cq += p * q;
            win_e += i * i + q * q;
        }

        if win_e <= 1e-12 {
            continue;
        }

        let denom = (pre_energy * win_e).sqrt().max(1e-12);
        let score = (ci * ci + cq * cq).sqrt() / denom;

        // Keep anything above a floor
        if score >= 0.20 {
            push_candidate(start, score, ci, cq);
        }
    }

    if best.is_empty() { None } else { Some(best) }
}

/// Compute a spectrogram (waterfall) of mono samples and save to a PNG.
/// - `samples`: mono PCM in [-1,1]
/// - `sample_rate`: sample rate in Hz
/// - `n_fft`: FFT size
/// - `hop`: hop size (overlap = n_fft - hop)
fn save_spectrogram_png(samples: &[f32], _sample_rate: u32, n_fft: usize, hop: usize, path: &str) {
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
fn read_stereo_wav(path: &str) -> (Vec<StereoSample>, WavSpec) {
    let mut reader = WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();

    assert_eq!(spec.channels, 2, "Input WAV must be stereo");

    println!(
        "[embed] Input WAV: {} Hz, {}-bit {:?}",
        spec.sample_rate, spec.bits_per_sample, spec.sample_format
    );

    let samples: Vec<StereoSample> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .chunks_exact(2)
            .map(|ch| {
                [
                    ch[0] as f32 / i16::MAX as f32,
                    ch[1] as f32 / i16::MAX as f32,
                ]
            })
            .collect(),
        (SampleFormat::Int, 24) => {
            const MAX_24: f32 = 8_388_607.0;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .chunks_exact(2)
                .map(|ch| [ch[0] as f32 / MAX_24, ch[1] as f32 / MAX_24])
                .collect()
        }
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .chunks_exact(2)
            .map(|ch| {
                [
                    ch[0] as f32 / i32::MAX as f32,
                    ch[1] as f32 / i32::MAX as f32,
                ]
            })
            .collect(),
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .chunks_exact(2)
            .map(|ch| [ch[0], ch[1]])
            .collect(),

        // More formats could be added, but the above are the common ones.
        _ => {
            panic!(
                "Unsupported WAV format: {:?} with {} bits per sample",
                spec.sample_format, spec.bits_per_sample
            );
        }
    };

    (samples, spec)
}

/// Write a stereo WAV file from Vec<[f32; 2]>.
fn write_stereo_wav(path: &str, samples: &[StereoSample], spec: WavSpec) {
    let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV file");

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            for &[l, r] in samples {
                let il = (l.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                let ir = (r.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                writer.write_sample(il).unwrap();
                writer.write_sample(ir).unwrap();
            }
        }
        (SampleFormat::Int, 24) => {
            const MAX_24: f32 = 8_388_607.0;
            for &[l, r] in samples {
                let il = (l.clamp(-1.0, 1.0) * MAX_24).round() as i32;
                let ir = (r.clamp(-1.0, 1.0) * MAX_24).round() as i32;
                writer.write_sample(il).unwrap(); // hound expects i32 container for 24-bit
                writer.write_sample(ir).unwrap();
            }
        }
        (SampleFormat::Int, 32) => {
            for &[l, r] in samples {
                let il = (l.clamp(-1.0, 1.0) * i32::MAX as f32).round() as i32;
                let ir = (r.clamp(-1.0, 1.0) * i32::MAX as f32).round() as i32;
                writer.write_sample(il).unwrap();
                writer.write_sample(ir).unwrap();
            }
        }
        (SampleFormat::Float, 32) => {
            for &[l, r] in samples {
                writer.write_sample(l.clamp(-1.0, 1.0)).unwrap();
                writer.write_sample(r.clamp(-1.0, 1.0)).unwrap();
            }
        }
        _ => panic!(
            "Unsupported output WAV format: {:?} {} bits",
            spec.sample_format, spec.bits_per_sample
        ),
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
    // --- 1) compute desired DSSS peak from dBFS ---
    let max_dsss = dsss
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| if v.abs() > m { v.abs() } else { m });

    let desired_peak = 10f32.powf(dsss_dbfs / 20.0);

    let mut scale = if max_dsss > 0.0 {
        desired_peak / max_dsss
    } else {
        0.0
    };

    // --- 2) headroom cap: ensure host + watermark won't clip ---
    // Find max absolute host amplitude per channel (or global)
    let mut host_peak = 0.0f32;
    for &[l, r] in audio {
        host_peak = host_peak.max(l.abs()).max(r.abs());
    }

    // How much room is left to hit ±1.0 (leave a tiny margin)
    let margin = 0.999f32;
    if host_peak >= margin {
        println!(
            "[mix] host audio already at/over margin (host_peak={} >= {}; requested dsss_dbfs={}); disabling DSSS injection to avoid further clipping",
            host_peak, margin, dsss_dbfs
        );
        scale = 0.0;
    }

    let headroom = (margin - host_peak).max(0.0);

    // DSSS injection peak after scaling will be max_dsss * scale
    // Cap scale so that (host_peak + injected_peak) <= margin
    if max_dsss > 0.0 {
        let max_scale_allowed = headroom / max_dsss;
        if scale > max_scale_allowed {
            println!(
                "[mix] capping DSSS scale due to headroom: requested_scale={} allowed_scale={} (host_peak={}, headroom={}, max_dsss={})",
                scale, max_scale_allowed, host_peak, headroom, max_dsss
            );
            scale = max_scale_allowed;
        }
    }

    println!(
        "[mix] max_dsss={} desired_peak={} final_scale={} host_peak={}",
        max_dsss, desired_peak, scale, host_peak
    );

    // --- 3) do the mix ---
    let mut out = Vec::with_capacity(audio.len());

    for (i, &[l0, r0]) in audio.iter().enumerate() {
        let d0 = if i < dsss.len() { dsss[i] * scale } else { 0.0 };

        let d1 = if i >= delay_samples {
            let j = i - delay_samples;
            if j < dsss.len() { dsss[j] * scale } else { 0.0 }
        } else {
            0.0
        };

        out.push([l0 + d0, r0 + d1]);
    }

    out
}

/// Generate a ±1 PRN code using ChaCha12Rng.
/// The seed can be any byte slice; it will be zero-padded
/// or truncated to 32 bytes (256 bits).
/// NOTE: ChaCha12 is used here purely as a deterministic PRNG
/// for spreading-code generation. This is NOT a cryptographic
/// security mechanism and provides no confidentiality.
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

/// Simple DSSS decoder
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
    ///
    /// Matched-filter despreader with *global* mean removal.
    /// (Avoid per-bit mean / per-bit energy normalization; they can destabilize slicing.)
    pub fn decode_from_chips(&self, selftest: bool, chips: &[f32]) -> Vec<u8> {
        let sf = self.config.spreading_factor;
        assert!(
            chips.len() % sf == 0,
            "chip stream length must be multiple of spreading_factor"
        );

        // --- Global bias removal (key fix) ---
        let mean = chips.iter().copied().sum::<f32>() / (chips.len().max(1) as f32);

        // Optional: cheap “how bad is the bias?” telemetry
        // (Leave this in while debugging; it’s very informative.)
        let mut e = 0.0f32;
        let mut peak = 0.0f32;
        for &v in chips {
            let x = v - mean;
            e += x * x;
            peak = peak.max(x.abs());
        }

        if selftest {
            let rms = (e / (chips.len().max(1) as f32)).sqrt();
            println!(
                "[despread] chips mean={:+.6e} rms={:.6e} peak={:.6e}",
                mean, rms, peak
            );
        }

        let prn = &self.config.prn_code;
        let n_bits = chips.len() / sf;
        let mut bits = Vec::with_capacity(n_bits);

        for bit_idx in 0..n_bits {
            let base = bit_idx * sf;

            // Matched filter / correlator
            let mut acc = 0.0f32;
            for k in 0..sf {
                let v = chips[base + k] - mean; // <-- apply global mean removal
                let pn = prn[k % prn.len()] as f32; // ±1
                acc += v * pn;
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

    /// Demodulate passband -> chips with coherent I/Q mixing + lowpass + integrate-and-dump,
    /// but allow an additional sample-phase offset (0..spc-1) for the integrate window.
    ///
    /// Also pads the tail by FIR group delay so the chip stream is not shortened by skip.
    ///
    /// Returns [chips_i, chips_q, -chips_i, -chips_q].
    pub fn demodulate_to_chips_nco_with_phase(
        &self,
        samples: &[f32],
        taps: usize,
        phase_samp: usize,
    ) -> [Vec<f32>; 4] {
        let spc = self.config.samples_per_chip.max(1);
        let sample_rate = self.config.sample_rate();
        let gd = (taps.saturating_sub(1)) / 2;

        // --- Tail pad so skip doesn't shorten chips (Finding A) ---
        let mut padded: Vec<f32> = Vec::with_capacity(samples.len() + gd);
        padded.extend_from_slice(samples);
        padded.extend(std::iter::repeat(0.0f32).take(gd));

        // 1) Mix down to I/Q
        let mut i_mix = Vec::with_capacity(padded.len());
        let mut q_mix = Vec::with_capacity(padded.len());

        let omega_dt = 2.0_f32 * PI * self.config.carrier_freq / sample_rate;
        let mut phase = 0.0f32;

        for &samp in &padded {
            let (sn, cs) = phase.sin_cos();
            i_mix.push(samp * cs);
            q_mix.push(samp * sn);

            phase += omega_dt;
            if phase > PI {
                phase -= 2.0 * PI;
            } else if phase < -PI {
                phase += 2.0 * PI;
            }
        }

        // 2) Lowpass filter I/Q around DC
        let lp = design_lowpass_fir(sample_rate, self.config.chip_rate * 0.75, taps);

        let i_filt = apply_fir(&i_mix, &lp);
        let q_filt = apply_fir(&q_mix, &lp);

        // 3) Skip group delay, align skip to chip boundary, then apply sample-phase
        let mut skip = gd;
        skip = ((skip + (spc - 1)) / spc) * spc;

        let phase_samp = phase_samp.min(spc.saturating_sub(1));
        let start = skip + phase_samp;

        let i_view = if i_filt.len() > start {
            &i_filt[start..]
        } else {
            &i_filt[..]
        };
        let q_view = if q_filt.len() > start {
            &q_filt[start..]
        } else {
            &q_filt[..]
        };

        // 4) Integrate-and-dump over each chip
        let n_chips = i_view.len() / spc;
        let mut chips_i = Vec::with_capacity(n_chips);
        let mut chips_q = Vec::with_capacity(n_chips);
        let mut chips_ni = Vec::with_capacity(n_chips);
        let mut chips_nq = Vec::with_capacity(n_chips);

        for (i_chunk, q_chunk) in i_view.chunks(spc).zip(q_view.chunks(spc)) {
            if i_chunk.len() < spc {
                break;
            }
            let acc_i: f32 = i_chunk.iter().sum::<f32>() / spc as f32;
            let acc_q: f32 = q_chunk.iter().sum::<f32>() / spc as f32;

            chips_i.push(acc_i);
            chips_q.push(acc_q);
            chips_ni.push(-acc_i);
            chips_nq.push(-acc_q);
        }

        [chips_i, chips_q, chips_ni, chips_nq]
    }

    /// Demodulate passband samples back down to DSSS chips (baseband),
    /// using coherent mixing + lowpass + integrate-and-dump.
    ///
    /// IMPORTANT: Pads the *tail* by FIR group delay so that skipping group delay
    /// does not shorten the resulting chip stream. This fixes the “missing 16 chips”
    /// failure in single-frame loopbacks.
    ///
    /// Returns [chips_i, chips_q, -chips_i, -chips_q].
    pub fn demodulate_to_chips_nco(&self, samples: &[f32], taps: usize) -> [Vec<f32>; 4] {
        let spc = self.config.samples_per_chip.max(1);
        let sample_rate = self.config.sample_rate();

        // FIR group delay in samples
        let gd = (taps.saturating_sub(1)) / 2;

        // --- Pad tail by group delay so skipping doesn't truncate the chip stream ---
        let mut padded: Vec<f32> = Vec::with_capacity(samples.len() + gd);
        padded.extend_from_slice(samples);
        padded.extend(std::iter::repeat(0.0f32).take(gd));

        // 1) Mix down to I/Q
        let mut i_mix = Vec::with_capacity(padded.len());
        let mut q_mix = Vec::with_capacity(padded.len());

        let omega_dt = 2.0_f32 * PI * self.config.carrier_freq / sample_rate;
        let mut phase = 0.0f32;

        for &samp in &padded {
            let (sn, cs) = phase.sin_cos();
            i_mix.push(samp * cs);
            q_mix.push(samp * sn);

            phase += omega_dt;
            if phase > PI {
                phase -= 2.0 * PI;
            } else if phase < -PI {
                phase += 2.0 * PI;
            }
        }

        // 2) Lowpass filter around DC
        let lp = design_lowpass_fir(sample_rate, self.config.chip_rate * 0.75, taps);

        let i_filt = apply_fir(&i_mix, &lp);
        let q_filt = apply_fir(&q_mix, &lp);

        // 3) Skip group delay, but align to a chip boundary
        let mut skip = gd;
        if spc > 1 {
            skip = ((skip + (spc - 1)) / spc) * spc;
        }

        let i_view = if i_filt.len() > skip {
            &i_filt[skip..]
        } else {
            &i_filt[..]
        };
        let q_view = if q_filt.len() > skip {
            &q_filt[skip..]
        } else {
            &q_filt[..]
        };

        // 4) Integrate-and-dump over each chip
        let n_chips = i_view.len() / spc;
        let mut chips_i = Vec::with_capacity(n_chips);
        let mut chips_q = Vec::with_capacity(n_chips);
        let mut chips_ni = Vec::with_capacity(n_chips);
        let mut chips_nq = Vec::with_capacity(n_chips);

        for (i_chunk, q_chunk) in i_view.chunks(spc).zip(q_view.chunks(spc)) {
            if i_chunk.len() < spc {
                break;
            }

            let acc_i: f32 = i_chunk.iter().sum::<f32>() / spc as f32;
            let acc_q: f32 = q_chunk.iter().sum::<f32>() / spc as f32;

            chips_i.push(acc_i);
            chips_q.push(acc_q);
            chips_ni.push(-acc_i);
            chips_nq.push(-acc_q);
        }

        [chips_i, chips_q, chips_ni, chips_nq]
    }

    /// Decode from a real passband signal assuming chip timing is
    /// already aligned.
    /// Fully noncoherent: select best of the four demod streams.
    pub fn decode_from_passband_no_preamble(
        &self,
        selftest: bool,
        samples: &[f32],
        taps: usize,
    ) -> Vec<u8> {
        // 1. Demodulate into 4 chip streams
        let chip_sets = self.demodulate_to_chips_nco(samples, taps);

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

        self.decode_from_chips(selftest, payload_chips)
    }

    fn unwrap_delta(mut d: f32) -> f32 {
        // map to (-pi, +pi]
        while d > std::f32::consts::PI {
            d -= 2.0 * std::f32::consts::PI;
        }
        while d <= -std::f32::consts::PI {
            d += 2.0 * std::f32::consts::PI;
        }
        d
    }

    /// Decode from passband using a sliding-window correlator against the known
    /// preamble. This can acquire timing even if the stream starts at an
    /// arbitrary position, e.g., after clipping.
    pub fn decode_from_passband_with_preamble(
        &self,
        selftest: bool,
        samples: &[f32],
        taps: usize,
    ) -> Option<Vec<u8>> {
        // Tunables
        let lattice_checks: usize = 3; // check k=1..=lattice_checks
        let k0_abs_thresh: f32 = 0.40; // k=0 must be strong enough
        let rel_frac: f32 = 0.25; // later hit >= rel_frac * k0
        let abs_floor: f32 = 0.12; // and never require < abs_floor
        let search_radius: usize = 64; // refine within +/- this many chips

        let require_hits_after_k0: usize = if samples.len() < 2 * frame_period_samples(&self.config)
        {
            0
        } else {
            1
        };

        if selftest {
            let tx_codeword_opt: Option<Vec<u8>> = std::fs::read("tx_codeword.bin").ok();
        }

        let spc = self.config.samples_per_chip.max(1);

        let pre_chips = preamble_chips(&self.config);
        let pre_len = pre_chips.len();
        let pre_e: f32 = pre_chips.iter().map(|x| x * x).sum();
        if pre_len == 0 || pre_e <= 1e-12 {
            return None;
        }

        let chips_per_bit = self.config.spreading_factor;
        let frame_period = frame_period_chips(&self.config);

        // Winner chosen by *highest signed preamble corr after phi rotation*
        // Store everything we need to finish decode without re-scanning candidates later.
        #[derive(Clone)]
        struct Winner {
            phase_samp: usize,
            start: usize,
            offset: usize,
            k0_sc: f32,
            signed_corr: f32,
            phi: f32,
            // we keep a copy of the coherently-combined chips for the winning phase
            chips: Vec<f32>,
            bit_err: Option<usize>,
        }

        let mut best: Option<Winner> = None;

        // ------------------------------------------------------------
        // Phase search: try integrate-and-dump starting offsets 0..spc-1
        // ------------------------------------------------------------
        for phase_samp in 0..spc {
            let chip_sets = self.demodulate_to_chips_nco_with_phase(samples, taps, phase_samp);
            let chips_i = &chip_sets[0];
            let chips_q = &chip_sets[1];

            if chips_i.len() < pre_len || chips_q.len() < pre_len {
                continue;
            }

            // Acquire candidates for this phase
            let candidates = match acquire_preamble_offset_iq(&pre_chips, chips_i, chips_q) {
                Some(v) => v,
                None => continue,
            };

            // scoring helper at exact start
            let score_at_iq = |start: usize| -> Option<(f32, f32, f32)> {
                if start + pre_len > chips_i.len() {
                    return None;
                }
                let mut ci = 0.0f32;
                let mut cq = 0.0f32;
                let mut win_e = 0.0f32;

                for k in 0..pre_len {
                    let i = chips_i[start + k];
                    let q = chips_q[start + k];
                    let p = pre_chips[k];
                    ci += p * i;
                    cq += p * q;
                    win_e += i * i + q * q;
                }
                if win_e <= 1e-12 {
                    return None;
                }

                let denom = (pre_e * win_e).sqrt().max(1e-12);
                let sc = (ci * ci + cq * cq).sqrt() / denom;
                Some((sc, ci, cq))
            };

            // refine helper near a center
            let best_near = |center: usize| -> Option<(usize, f32, f32, f32)> {
                let max_start = chips_i.len().saturating_sub(pre_len);
                let lo = center.saturating_sub(search_radius);
                let hi = (center + search_radius).min(max_start);

                let mut best_local: Option<(usize, f32, f32, f32)> = None;
                for s in lo..=hi {
                    if let Some((sc, ci, cq)) = score_at_iq(s) {
                        if best_local.map_or(true, |b| sc > b.1) {
                            best_local = Some((s, sc, ci, cq));
                        }
                    }
                }
                best_local
            };

            // ------------------------------------------------------------
            // Lattice-verify all candidates for this phase
            // ------------------------------------------------------------
            for (cand_off, _cand_sc, _cand_ci, _cand_cq) in candidates {
                // refine candidate to best local peak
                let (offset, k0_sc, ci0, cq0) = match best_near(cand_off) {
                    Some(v) => v,
                    None => continue,
                };

                if k0_sc < k0_abs_thresh {
                    continue;
                }

                let kthresh = (rel_frac * k0_sc).max(abs_floor);

                let mut hits_after = 0usize;

                for k in 1..=lattice_checks {
                    let expect = offset + k * frame_period;
                    if let Some((_idx, sc, _ci, _cq)) = best_near(expect) {
                        if sc >= kthresh {
                            hits_after += 1;
                        }
                    }
                }

                if hits_after < require_hits_after_k0 {
                    continue;
                }

                // ------------------------------------------------------------
                // For lattice-passing candidate: estimate phi & coherent combine
                // then compute signed preamble correlation, which is our ranking metric.
                // ------------------------------------------------------------
                let phi = cq0.atan2(ci0);
                let phi0 = phi;

                // Also try to estimate phi at k=1 (or the strongest later hit)
                let mut phi1_opt: Option<(usize, f32)> = None; // (chip_index, phi)

                for k in 1..=lattice_checks {
                    let expect = offset + k * frame_period;
                    if let Some((idx, sc, ci, cq)) = best_near(expect) {
                        if sc >= kthresh {
                            let phi = cq.atan2(ci);
                            // choose the earliest good one (k=1 usually)
                            phi1_opt = Some((idx, phi));
                            break;
                        }
                    }
                }

                let (phi1_chip_idx, phi1) = match phi1_opt {
                    Some(v) => v,
                    None => continue, // or allow drift=0, but better to insist for now
                };

                let delta = DsssDecoder::unwrap_delta(phi1 - phi0);
                let dphi_per_chip =
                    delta / ((phi1_chip_idx as isize - offset as isize).abs().max(1) as f32);

                println!(
                    "[acq] phase={} drift: phi0={:+.3} phi1={:+.3} dphi/chip={:+.3e}",
                    phase_samp, phi0, phi1, dphi_per_chip
                );

                let dphi = dphi_per_chip;

                let chips: Vec<f32> = chips_i
                    .iter()
                    .zip(chips_q.iter())
                    .enumerate()
                    .map(|(n, (&i, &q))| {
                        // phase relative to the preamble start (offset)
                        let rel = n as f32 - offset as f32;
                        let phi = phi0 + dphi * rel;
                        let (c, s) = (phi.cos(), phi.sin());
                        i * c + q * s
                    })
                    .collect();

                // signed normalized correlation on preamble at offset
                let mut acc = 0.0f32;
                let mut win_e = 0.0f32;
                for k in 0..pre_len {
                    let v = chips[offset + k];
                    acc += pre_chips[k] * v;
                    win_e += v * v;
                }
                if win_e <= 1e-12 {
                    continue;
                }
                let signed_corr = acc / (pre_e * win_e).sqrt().max(1e-12);

                if signed_corr > 0.3 {
                    println!(
                        "[acq] phase={} best: signed_corr={:+.4} k0_sc={:.4} offset={} phi={:+.3}",
                        phase_samp, signed_corr, k0_sc, offset, phi
                    );
                }

                let needed = RS_CODEWORD_LEN * 8 * chips_per_bit;

                let base_start = offset + pre_len;

                // Small chip-domain refinement around payload start. This is the likely culprit
                // when preamble locks but codeword is garbage.
                let delta_max: isize = (2 * chips_per_bit) as isize; // try ±2 bits worth of chips
                let mut best_start: Option<(usize, Option<usize>, bool)> = None; // (start, bit_err, rs_ok)

                for delta in (-delta_max)..=delta_max {
                    let s = base_start as isize + delta;
                    if s < 0 {
                        continue;
                    }
                    let s = s as usize;
                    if s + needed > chips.len() {
                        continue;
                    }

                    let payload_chips = &chips[s..s + needed];
                    let bytes = self.decode_from_chips(selftest, payload_chips);
                    if bytes.len() < RS_CODEWORD_LEN {
                        continue;
                    }

                    let bit_err = if let Some(tx) = &tx_codeword_opt {
                        let rx = &bytes[..RS_CODEWORD_LEN];
                        let tx = &tx[..tx.len().min(RS_CODEWORD_LEN)];
                        if tx.len() == RS_CODEWORD_LEN {
                            Some(ber_bytes(tx, rx).2)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // If we don't have TX for BER, use "RS correct() succeeds" as the objective.
                    let rs_ok = if tx_codeword_opt.is_none() {
                        let mut cw = [0u8; RS_CODEWORD_LEN];
                        cw.copy_from_slice(&bytes[..RS_CODEWORD_LEN]);
                        let dec = Decoder::new(RS_ECC_LEN);
                        dec.correct(&mut cw, None).is_ok()
                    } else {
                        false
                    };

                    let better = match (&best_start, bit_err, rs_ok) {
                        // Prefer RS success if that's the only available objective
                        (Some((_bs, _be, true)), _, false) => false,
                        (Some((_bs, _be, false)), _, true) => true,

                        // Prefer lower BER if both have it
                        (Some((_bs, Some(prev), _)), Some(cur), _) => cur < *prev,

                        // Prefer any BER-known candidate over unknown
                        (Some((_bs, None, _)), Some(_), _) => true,

                        // Otherwise keep first viable (or you can fall back to signed_corr outside)
                        (None, _, _) => true,
                        _ => false,
                    };

                    if better {
                        best_start = Some((s, bit_err, rs_ok));
                    }
                }

                if let Some((start, bit_err, rs_ok)) = best_start {
                    // Optional: light telemetry so you can see the start refinement working
                    if rs_ok {
                        println!(
                            "[acq] phase={} cand offset={} start={} (δ={}) RS_OK",
                            phase_samp,
                            offset,
                            start,
                            start as isize - base_start as isize
                        );
                    }

                    let is_better = match (&best, bit_err) {
                        // Prefer lower BER if both have it
                        (Some(b), Some(be)) if b.bit_err.is_some() => be < b.bit_err.unwrap(),

                        // Prefer any BER-known candidate over unknown
                        (Some(b), Some(_)) if b.bit_err.is_none() => true,

                        // Otherwise fall back to signed preamble correlation
                        _ => best.as_ref().map_or(true, |b| signed_corr > b.signed_corr),
                    };
                    if is_better {
                        best = Some(Winner {
                            phase_samp,
                            offset,
                            start,
                            k0_sc,
                            signed_corr,
                            phi,
                            chips,
                            bit_err,
                        });
                    }

                    if let Some(be) = bit_err {
                        println!(
                            "[acq] phase={} cand offset={} signed_corr={:+.4} BER_bits={}/{}",
                            phase_samp,
                            offset,
                            signed_corr,
                            be,
                            RS_CODEWORD_LEN * 8
                        );
                    }
                }
            }
        }

        let best = best?;
        println!(
            "[decode] phase={} chosen offset={} k0_sc={} phi={} signed_pre_corr={}",
            best.phase_samp, best.offset, best.k0_sc, best.phi, best.signed_corr
        );

        // If signed corr is negative, flip polarity (π ambiguity)
        let mut chips = best.chips;
        if best.signed_corr < 0.0 {
            for v in &mut chips {
                *v = -*v;
            }
            println!(
                "[decode] flipped chip polarity (signed pre corr = {})",
                best.signed_corr
            );
        } else {
            println!(
                "[decode] chip polarity OK (signed pre corr = {})",
                best.signed_corr
            );
        }

        // ------------------------------------------------------------
        // Extract payload chips and decode
        // ------------------------------------------------------------
        let start = best.start;
        let needed = RS_CODEWORD_LEN * 8 * chips_per_bit;

        println!(
            "[decode] chips.len()={} start={} needed={}",
            chips.len(),
            start,
            needed
        );

        if start + needed > chips.len() {
            eprintln!("[decode] late lock — insufficient chips");
            return None;
        }

        let payload_chips = &chips[start..start + needed];
        let bytes = self.decode_from_chips(selftest, payload_chips);
        println!(
            "[decode] despread bytes.len()={} (need >= {})",
            bytes.len(),
            RS_CODEWORD_LEN
        );

        if bytes.len() >= RS_CODEWORD_LEN {
            // quick sanity: how "structured" is the first block?
            let head = &bytes[..16];
            let ones: u32 = head.iter().map(|b| b.count_ones()).sum();
            println!("[decode] first16 ones={} ({:02X?})", ones, head);
        }

        if selftest {
            std::fs::write(
                "rx_codeword.bin",
                &bytes[..bytes.len().min(RS_CODEWORD_LEN)],
            )
            .ok();

            // If a tx_codeword.bin is present, print BER stats
            if let Ok(tx) = std::fs::read("tx_codeword.bin") {
                let rx = &bytes[..bytes.len().min(RS_CODEWORD_LEN)];
                let tx = &tx[..tx.len().min(RS_CODEWORD_LEN)];
                let (n, byte_err, bit_err) = ber_bytes(tx, rx);
                println!(
                    "[diag] codeword compare: n={} byte_err={}/{} bit_err={}/{}",
                    n,
                    byte_err,
                    n,
                    bit_err,
                    n * 8
                );
            }
        }

        if bytes.len() < RS_CODEWORD_LEN {
            eprintln!("[decode] FAIL: despread produced too few bytes");
            return None;
        }

        // ------------------------------------------------------------
        // RS decode (unchanged)
        // ------------------------------------------------------------
        let mut codeword = [0u8; RS_CODEWORD_LEN];
        codeword.copy_from_slice(&bytes[..RS_CODEWORD_LEN]);

        let dec = Decoder::new(RS_ECC_LEN);
        let recovered = match dec.correct(&mut codeword, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[decode] FAIL: RS correct() error: {:?}", e);
                return None;
            }
        };

        let data = recovered.data();
        if data.len() < 4 {
            eprintln!("[decode] FAIL: RS data too short for length header");
            return None;
        }

        let payload_len = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
        println!("[decode] payload_len={}", payload_len);

        if payload_len > RS_DATA_LEN - 4 {
            eprintln!("[decode] FAIL: payload_len out of range");
            return None;
        }

        Some(data[4..4 + payload_len].to_vec())
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

    let taps = 257; // or 513 for sharper skirts; no need seen experimentally
    let spreading_factor = args.spreading_factor;
    let carrier_freq = args.carrier_freq;
    let samples_per_chip = args.samples_per_chip;

    // Use ChaCha12 to produce a PRNG spreading code.
    // PRN depends only on spreading_factor + seed, not on chip_rate / Fs
    let prn_code = generate_prn_chacha12(spreading_factor, args.seed.as_bytes());

    let make_config = |chip_rate: f32| DsssConfig {
        spreading_factor,
        chip_rate,
        carrier_freq,
        samples_per_chip,
        prn_code: prn_code.clone(), // cheap-ish; Vec<i8> clone
    };

    let config = make_config(args.chip_rate);

    //----------------------------------------------------------
    // MODE: ENCODE
    //----------------------------------------------------------
    if matches!(args.mode, Mode::Encode) {
        println!("[encode] reading plaintext message…");

        let encoder = DsssEncoder::new(config.clone());

        let msg_bytes = args.message.as_bytes();

        println!("[encode] encoding {} bytes…", msg_bytes.len());
        let tx_samples = encoder.encode_to_passband(msg_bytes);

        println!(
            "[encode] writing {} float samples to {}",
            tx_samples.len(),
            args.output
        );

        // Write raw f32 samples (little-endian)
        let mut f = File::create(&args.output).expect("failed to create output file");
        let mut buf = Vec::with_capacity(tx_samples.len() * 4);
        for &s in &tx_samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&buf).expect("failed to write samples");

        println!("[encode] done.");
        return;
    }

    //----------------------------------------------------------
    // MODE: DECODE
    //----------------------------------------------------------
    if matches!(args.mode, Mode::Decode) {
        println!("[decode] reading passband samples from {}", args.input);

        let decoder = DsssDecoder::new(config.clone());

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
        let recovered = decoder.decode_from_passband_with_preamble(args.selftest, &samples, taps);

        // Write decoded bytes to output file
        match recovered {
            Some(ref bytes) => {
                println!(
                    "[debug] recovered {} bytes, writing to {}",
                    bytes.len(),
                    args.output
                );

                let mut f =
                    std::fs::File::create(&args.output).expect("failed to create output file");
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
        let (audio, input_spec) = read_stereo_wav(&args.input);
        let wav_fs_f = input_spec.sample_rate as f32;

        //--------------------------------------------------
        // 2. Auto-deduce chip_rate from sample rate
        //--------------------------------------------------
        let spc = args.samples_per_chip.max(1) as f32;
        let chip_rate = wav_fs_f / spc;

        println!(
            "[embed] Input Fs = {} Hz, auto chip_rate = {} chips/s, samples_per_chip = {}",
            input_spec.sample_rate, chip_rate, args.samples_per_chip
        );

        //--------------------------------------------------
        // 3. Enforce Nyquist: carrier_freq < Fs/2
        //--------------------------------------------------
        let nyquist = wav_fs_f / 2.0;
        if args.carrier_freq <= 0.0 || args.carrier_freq >= nyquist {
            panic!(
                "carrier_freq={} Hz is invalid for Fs={} Hz (Nyquist = {} Hz). \
                 Choose 0 < carrier_freq < Nyquist.",
                args.carrier_freq, input_spec.sample_rate, nyquist
            );
        }

        //--------------------------------------------------
        // 4. Build PRN & DSSS config
        //--------------------------------------------------
        let chip_rate = input_spec.sample_rate as f32 / args.samples_per_chip.max(1) as f32;
        let config = make_config(chip_rate);
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
        if selftest {
            std::fs::write("tx_codeword.bin", rs_codeword)
                .expect("failed to write tx_codeword.bin");
        }

        // RS SELF-TEST (no DSSS at all)
        if args.selftest {
            println!(
                "[debug] RS self-test: data.len() = {}, codeword.len() = {}",
                rs_data.len(),
                rs_codeword.len()
            );

            let mut cw = rs_codeword.to_vec();
            let dec = Decoder::new(RS_ECC_LEN);

            match dec.correct(&mut cw, None) {
                Ok(recovered) => {
                    let data_back = recovered.data();
                    println!(
                        "[debug] RS self-test: recovered.data().len() = {}",
                        data_back.len()
                    );

                    if data_back == &rs_data[..] {
                        println!("[debug] RS self-test: SUCCESS (data matches)");
                    } else {
                        println!("[debug] RS self-test: MISMATCH!");
                        println!(
                            "[debug] orig[0..32]:   {:02X?}",
                            &rs_data[..32.min(rs_data.len())]
                        );
                        println!(
                            "[debug] recov[0..32]: {:02X?}",
                            &data_back[..32.min(data_back.len())]
                        );
                    }
                }
                Err(e) => {
                    println!("[debug] RS self-test: FAILED with {:?}", e);
                }
            }
        }

        //--------------------------------------------------
        // 6.3. Build DSSS frame bits: PREAMBLE_BITS + bits(rs_codeword)
        //--------------------------------------------------
        let mut frame_bits = Vec::with_capacity(frame_bits_len());
        frame_bits.extend_from_slice(PREAMBLE_BITS);
        frame_bits.extend(bytes_to_bits(rs_codeword));

        debug_assert_eq!(
            frame_bits.len(),
            frame_bits_len(),
            "frame_bits length drifted from frame_bits_len()"
        );

        //--------------------------------------------------
        // 6.4. As before: bits -> chips -> passband
        //--------------------------------------------------
        let frame_chips = bits_to_chips(&frame_bits, &config);
        let frame = chips_to_passband(&frame_chips, &config);

        debug_assert_eq!(
            frame.len(),
            frame_period_samples(&config),
            "passband frame length mismatch"
        );

        // println!(
        //     "[debug] passband frame has {} samples ({} chips)",
        //     frame.len(),
        //     frame_period_chips(&config)
        // );

        let rt_chips = decoder.demodulate_to_chips_nco(&frame, taps);
        // println!(
        //     "[debug] passband roundtrip: frame_chips.len()={}, rt_chips[0].len()={}",
        //     frame_chips.len(),
        //     rt_chips[0].len()
        // );

        let n_show = frame_chips.len().min(rt_chips[0].len()).min(16);

        // Compare first few chips numerically
        if args.selftest {
            for i in 0..n_show {
                println!(
                    "[debug] chip[{}]: orig={:.4} rt={:.4}",
                    i, frame_chips[i], rt_chips[0][i]
                );
            }
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
        println!(
            "[debug] partial chip correlation (first {}): {}",
            n_show, corr
        );

        // DEBUG: loopback test of DSSS+RS without audio channel
        if args.selftest {
            println!("[debug] running DSSS loopback self-test…");

            // "Transmit" frame directly
            let tx = frame.clone();

            // "Receive" and decode directly
            let recovered = decoder.decode_from_passband_with_preamble(true, &tx, taps);
            let raw_bytes = decoder.decode_from_passband_no_preamble(true, &tx, taps);
            println!(
                "[debug] first 32 bytes (no preamble): {:02X?}",
                &raw_bytes[..32.min(raw_bytes.len())]
            );

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
        let frame_len = frame_period_samples(&config);

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
        let mut dsss_stream = vec![0.0_f32; audio_len];
        for f in 0..n_frames {
            let start = f * frame_len;
            let end = start + frame_len;
            if end > audio_len {
                break;
            }
            dsss_stream[start..end].copy_from_slice(&frame[..(end - start)]);
        }

        println!(
            "[embed] DSSS frame length = {} samples, frames embedded = {}, total DSSS len = {}",
            frame_len,
            n_frames,
            dsss_stream.len()
        );

        //--------------------------------------------------
        // 7. Compute chip-aligned delay between L & R in samples
        //--------------------------------------------------
        // frame_len is in samples; convert to chips (should be exact here)

        let frame_len_samples = frame_len;
        let spc_u = args.samples_per_chip.max(1);
        debug_assert!(
            frame_len_samples % spc_u == 0,
            "frame_len_samples must be an integer number of chips (frame_len_samples={} spc={})",
            frame_len_samples,
            spc_u
        );

        let frame_len_chips = frame_len_samples / spc_u;

        // Delay in chips, then convert to samples
        let delay_fraction = args.delay_fraction.clamp(0.0, 1.0);
        let delay_chips = ((frame_len_chips as f32) * delay_fraction).round() as usize;

        // ensure within bounds (optional safety)
        let delay_chips = delay_chips.min(frame_len_chips.saturating_sub(1));
        let delay_samples = delay_chips * spc_u;

        println!(
            "[embed] Using frame_len_samples={} frame_len_chips={} delay_fraction={:.3} → delay_chips={} → delay_samples={}",
            frame_len_samples, frame_len_chips, delay_fraction, delay_chips, delay_samples
        );

        //--------------------------------------------------
        // 8. Mix DSSS into stereo audio
        //--------------------------------------------------
        let mixed = mix_dsss_stereo_delayed(&audio, &dsss_stream, delay_samples, args.dsss_dbfs);

        let original_left: Vec<f32> = audio.iter().map(|[l, _]| *l).collect();
        let mixed_left: Vec<f32> = mixed.iter().map(|[l, _]| *l).collect();
        let original_right: Vec<f32> = audio.iter().map(|[_, r]| *r).collect();
        let mixed_right: Vec<f32> = mixed.iter().map(|[_, r]| *r).collect();

        if args.visualize {
            save_spectrogram_png(
                &original_left,
                input_spec.sample_rate,
                1024, // n_fft
                512,  // hop
                "orig_left.png",
            );

            save_spectrogram_png(
                &mixed_left,
                input_spec.sample_rate,
                1024,
                512,
                "mixed_left.png",
            );

            save_spectrogram_png(
                &original_right,
                input_spec.sample_rate,
                1024, // n_fft
                512,  // hop
                "orig_right.png",
            );

            save_spectrogram_png(
                &mixed_right,
                input_spec.sample_rate,
                1024,
                512,
                "mixed_right.png",
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
        write_stereo_wav(&args.output, &mixed, input_spec);
        println!("[embed] Wrote embedded audio to '{}'", args.output);
        return;
    }

    //----------------------------------------------------------
    // MODE: DECODE-WAV  (decode DSSS from a stereo WAV file)
    //----------------------------------------------------------
    if matches!(args.mode, Mode::DecodeWav) {
        // 1. Read stereo WAV and derive sample rate
        let (audio, input_spec) = read_stereo_wav(&args.input);
        let wav_fs_f = input_spec.sample_rate as f32;

        // 2. Auto-chip-rate from Fs and samples_per_chip (same as EMBED)
        let spc = args.samples_per_chip.max(1) as f32;
        let chip_rate = wav_fs_f / spc;

        println!(
            "[decode-wav] Input Fs = {} Hz, auto chip_rate = {} chips/s, samples_per_chip = {}",
            input_spec.sample_rate, chip_rate, args.samples_per_chip
        );

        // 3. Nyquist safety for carrier
        let nyquist = wav_fs_f / 2.0;
        if args.carrier_freq <= 0.0 || args.carrier_freq >= nyquist {
            panic!(
                "carrier_freq={} Hz is invalid for Fs={} Hz (Nyquist = {} Hz). \
                 Choose 0 < carrier_freq < Nyquist.",
                args.carrier_freq, input_spec.sample_rate, nyquist
            );
        }

        // 4. Rebuild PRN & DSSS config to match EMBED
        let config = make_config(chip_rate);
        let decoder = DsssDecoder::new(config);

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

        println!("[decode-wav] mono stream has {} samples", mono.len());

        let filtered = mono;

        if args.selftest {
            let chips = decoder.demodulate_to_chips_nco(&filtered, taps);

            for i in 0..16 {
                println!("[debug] chip[{}] = {}", i, chips[0][i]);
            }
        }

        // 6. Run preamble-based sliding-window decode
        match decoder.decode_from_passband_with_preamble(args.selftest, &filtered, taps) {
            Some(bytes) => {
                println!(
                    "[decode-wav] recovered {} bytes, writing to {}",
                    bytes.len(),
                    args.output
                );
                std::fs::write(&args.output, &bytes).expect("failed to write decoded payload");
            }
            None => {
                eprintln!(
                    "[decode-wav] failed: no valid frame recovered (see RS / preamble logs above)"
                );
                std::process::exit(1);
            }
        }

        return;
    }

    unreachable!("mode must be encode, decode, embed, or decode-wav");
}
