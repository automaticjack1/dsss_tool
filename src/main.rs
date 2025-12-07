use clap::{ValueEnum, Parser};
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use image::{GrayImage, Luma};
use rand_chacha::ChaCha12Rng;
use rand::{Rng, SeedableRng};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{Read, Write};

type StereoSample = [f32; 2];

// Example: fixed preamble bits (Barker-like or just pseudo-random)
const PREAMBLE_BITS: &[u8] = &[
    1,0,1,1,0,1,0,0,  1,1,0,0,1,0,1,1,
    0,1,0,1,1,0,0,1,  1,0,1,0,0,1,1,0,
];

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

fn bits_to_chips(bits: &[u8], cfg: &DsssConfig) -> Vec<f32> {
    let mut chips = Vec::with_capacity(bits.len() * cfg.spreading_factor);
    for &bit in bits {
        let bit_val = if bit == 1 { 1.0 } else { -1.0 };
        for j in 0..cfg.spreading_factor {
            let code_val = cfg.prn_code[j % cfg.prn_code.len()] as f32;
            chips.push(bit_val * code_val);
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
    preamble_chips: &[f32],
    chips: &[f32],
    max_search: usize,
) -> Option<usize> {
    let n_p = preamble_chips.len();
    if n_p == 0 || chips.len() < n_p {
        return None;
    }

    let limit = chips.len().saturating_sub(n_p).min(max_search);

    let mut best_offset = None;
    let mut best_metric = 0.0_f32;

    for k in 0..=limit {
        let mut acc = 0.0_f32;
        for i in 0..n_p {
            acc += chips[k + i] * preamble_chips[i];
        }
        let metric = acc.abs();
        if metric > best_metric {
            best_metric = metric;
            best_offset = Some(k);
        }
    }

    best_offset
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

        let mut bits = Vec::with_capacity(chips.len() / n);

        for symbol_chips in chips.chunks(n) {
            let mut acc = 0.0_f32;
            for (j, &chip) in symbol_chips.iter().enumerate() {
                let code_val = self.config.prn_code[j % self.config.prn_code.len()] as f32;
                acc += chip * code_val;
            }

            let bit = if acc >= 0.0 { 1 } else { 0 };
            bits.push(bit);
        }

        bits_to_bytes(&bits)
    }

    /// Demodulate passband samples back down to DSSS chips (baseband).
    ///
    /// This performs a coherent demodulation against the configured carrier and
    /// integrates over each chip interval. Any trailing partial chip is ignored.
    pub fn demodulate_to_chips(&self, samples: &[f32]) -> Vec<f32> {
        let spc = self.config.samples_per_chip;
        let sample_rate = self.config.sample_rate();
        let dt = 1.0_f32 / sample_rate;
        let omega_c = 2.0_f32 * PI * self.config.carrier_freq;

        let mut chips = Vec::with_capacity(samples.len() / spc.max(1));

        // Coherent demod: multiply by cos(ωc t) and integrate over each chip.
        let mut t0 = 0.0_f32;
        for chip_samples in samples.chunks(spc) {
            if chip_samples.len() < spc {
                break; // drop any partial chip at the end
            }

            let mut acc = 0.0_f32;
            for (i, &s) in chip_samples.iter().enumerate() {
                let t = t0 + (i as f32) * dt;
                let carrier = (omega_c * t).cos();
                acc += s * carrier;
            }

            t0 += (spc as f32) * dt;
            chips.push(acc);
        }

        chips
    }

    /// Decode from a real passband signal at the configured carrier frequency,
    /// assuming chip timing is already aligned.
    pub fn decode_from_passband(&self, samples: &[f32]) -> Vec<u8> {
        let chips = self.demodulate_to_chips(samples);
        self.decode_from_chips(&chips)
    }

    /// Decode from passband using a sliding-window correlator against the known
    /// preamble. This can acquire timing even if the stream starts at an
    /// arbitrary position, e.g., after clipping.
    pub fn decode_from_passband_with_preamble(&self, samples: &[f32]) -> Option<Vec<u8>> {
        let chips = self.demodulate_to_chips(samples);
        let pre_chips = preamble_chips(&self.config);

        if chips.len() < pre_chips.len() {
            return None;
        }

        let max_search = chips.len().saturating_sub(pre_chips.len());
        let offset = acquire_preamble_offset(&pre_chips, &chips, max_search)?;

        println!(
            "[decode] acquired preamble at chip offset {} (of {})",
            offset,
            chips.len()
        );

        let start = offset + pre_chips.len();
        if start >= chips.len() {
            return None;
        }

        let payload_chips = &chips[start..];

        // snap to full symbols
        let n = self.config.spreading_factor;
        let usable_len = (payload_chips.len() / n) * n;
        let payload_chips = &payload_chips[..usable_len];

        let bytes = self.decode_from_chips(payload_chips);
        if bytes.len() < 4 {
            eprintln!("[decode] too few bytes for length header");
            return None;
        }

        let len = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        if bytes.len() < 4 + len {
            eprintln!(
                "[decode] declared length {} but only {} bytes available",
                len,
                bytes.len().saturating_sub(4)
            );
            // You can either treat this as error, or return whatever you have:
            // return None;
        }

        let end = (4 + len).min(bytes.len());
        let msg = bytes[4..end].to_vec();

        Some(msg)
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

/// Utility: convert bits (MSB-first) back to bytes.
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((bits.len() + 7) / 8);
    let mut current_byte = 0u8;
    for (i, &bit) in bits.iter().enumerate() {
        current_byte = (current_byte << 1) | (bit & 1);
        if i % 8 == 7 {
            bytes.push(current_byte);
            current_byte = 0;
        }
    }
    let rem = bits.len() % 8;
    if rem != 0 {
        let shift = 8 - rem;
        bytes.push(current_byte << shift);
    }
    bytes
}

fn main() {
    let args = Args::parse();

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
    let decoder = DsssDecoder::new(config);

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
        let recovered = decoder.decode_from_passband(&samples);

        println!("[decode] decoded {} bytes. Writing to {}",
                 recovered.len(),
                 args.output);

        // Write decoded bytes to output file
        let mut f = File::create(&args.output)
        .expect("failed to create output file");
        f.write_all(&recovered).expect("failed to write output message");

        println!(
            "[decode] message recovered: {}",
            String::from_utf8_lossy(&recovered)
        );

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
        let payload_len = payload.len() as u32;
        let mut frame_bytes = Vec::with_capacity(4 + payload.len());
        frame_bytes.extend_from_slice(&payload_len.to_be_bytes());
        frame_bytes.extend_from_slice(&payload);

        let mut frame_bits = Vec::with_capacity(
            PREAMBLE_BITS.len() + frame_bytes.len() * 8
        );
        frame_bits.extend_from_slice(PREAMBLE_BITS);
        frame_bits.extend(bytes_to_bits(&frame_bytes));

        let frame_chips = bits_to_chips(&frame_bits, &config);
        let frame = chips_to_passband(&frame_chips, &config);

        let audio_len = audio.len();
        let frame_len = frame.len();

        let n_frames = if frame_len == 0 {
            0
        } else {
            audio_len / frame_len
        };

        if n_frames == 0 {
            println!(
                "[embed] Warningaudio too short for a single DSSS frame ({} samples, need {}). No DSSS will be embedded.",
                audio_len, frame_len
            );
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
    
        println!(
            "[decode-wav] mono stream has {} samples",
            mono.len()
        );
    
        // 6. Run preamble-based sliding-window decode
        match decoder.decode_from_passband_with_preamble(&mono) {
            Some(bytes) => {
                println!(
                    "[decode-wav] recovered {} bytes, writing to {}",
                    bytes.len(),
                    args.output
                );
                std::fs::write(&args.output, &bytes)
                    .expect("failed to write decoded payload");
    
                //println!(
                    //"[decode-wav] as UTF-8 (lossy): {}",
                    //String::from_utf8_lossy(&bytes)
                //);
            }
            None => {
                eprintln!("[decode-wav] failed: no preamble found or payload too short");
            }
        }
    
        return;
    }
    
    unreachable!("mode must be encode, decode, or embed");
}
    
