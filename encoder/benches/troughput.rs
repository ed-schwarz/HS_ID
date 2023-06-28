use criterion::*;
use encoder::{Algorithm, Encoder, SymbolSize};

use dactyl::total_cmp;
use std::time::Instant;
use std::{cmp::Ordering, time::Duration};

#[derive(Debug, Clone, Copy)]
/// # Error.
///
/// This enum serves as the custom error type for `Brunch`.
pub enum BrunchError {
    /// # Duplicate name.
    DupeName,

    /// # No benches were specified.
    NoBench,

    /// # A bench was missing a [`Bench::run`](crate::Bench::run)-type call.
    NoRun,

    /// # General math failure. (Floats aren't fun.)
    Overflow,

    /// # The benchmark completed too quickly to analyze.
    TooFast,

    /// # Not enough samples were collected to analyze.
    TooSmall(u32),

    /// # The samples were too chaotic to analyze.
    TooWild,
}

#[derive(Debug, Clone, Copy)]
/// # Runtime Stats!
pub(crate) struct Stats {
    /// # Total Samples.
    total: u32,

    /// # Valid Samples.
    valid: u32,

    /// # Standard Deviation.
    deviation: f64,

    /// # Mean Duration of Valid Samples.
    mean: f64,
}

impl TryFrom<Vec<Duration>> for Stats {
    type Error = BrunchError;
    fn try_from(samples: Vec<Duration>) -> Result<Self, Self::Error> {
        let total = u32::try_from(samples.len()).unwrap_or(u32::MAX);
        if total < MIN_SAMPLES {
            return Err(BrunchError::TooSmall(total));
        }

        // Crunch!
        let mut calc = Abacus::from(samples);
        calc.prune_outliers();

        let valid = u32::try_from(calc.len()).unwrap_or(u32::MAX);
        if valid < MIN_SAMPLES {
            return Err(BrunchError::TooWild);
        }

        let mean = calc.mean();
        let deviation = calc.deviation();

        // Done!
        let out = Self { total, valid, deviation, mean };
        if out.is_valid() {
            Ok(out)
        } else {
            Err(BrunchError::Overflow)
        }
    }
}
impl Stats {
    /// # Is Valid?
    fn is_valid(self) -> bool {
        MIN_SAMPLES <= self.valid
            && self.valid <= self.total
            && self.deviation.is_finite()
            && total_cmp!((self.deviation) >= 0.0)
            && self.mean.is_finite()
            && total_cmp!((self.mean) >= 0.0)
    }
}

#[derive(Debug)]
/// # Abacus.
///
/// This struct wraps a set of durations (from i.e. a bench run), providing
/// methods to calculate relevant metrics like mean, standard deviation,
/// quantiles, etc.
///
/// (This is basically where the stats from `Stats` come from.)
pub(crate) struct Abacus {
    set: Vec<f64>,
    len: usize,
    unique: usize,
    total: f64,
}

impl From<Vec<Duration>> for Abacus {
    fn from(src: Vec<Duration>) -> Self {
        let set: Vec<f64> = src.iter().map(Duration::as_secs_f64).collect();
        Self::from(set)
    }
}

impl From<Vec<f64>> for Abacus {
    fn from(mut set: Vec<f64>) -> Self {
        // Negative and abnormal values make no sense for our purposes, so
        // let's pre-emptively strip them out.
        set.retain(|f| match f.total_cmp(&0.0) {
            Ordering::Equal => true,
            Ordering::Greater if f.is_normal() => true,
            _ => false,
        });

        // Everything from here on out requires a sorted set, so let's take
        // care of that now.
        set.sort_by(f64::total_cmp);

        // Pre-calculate some useful totals.
        let len = set.len();
        let unique = count_unique(&set);
        let total = set.iter().sum();

        // Done!
        Self { set, len, unique, total }
    }
}

impl Abacus {
    /// # Is Empty?
    const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// # Length.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    #[allow(clippy::cast_precision_loss)]
    /// # Float Length.
    const fn f_len(&self) -> f64 {
        self.len as f64
    }
}

impl Abacus {
    /// # Standard Deviation.
    ///
    /// Note: this uses the _n_ rather than _n+1_ approach.
    pub(crate) fn deviation(&self) -> f64 {
        if self.is_empty() || self.unique == 1 {
            return 0.0;
        }
        let mean = self.mean();
        let squares: Vec<f64> = self.set.iter().map(|n| (mean - *n).powi(2)).collect();
        let sum: f64 = squares.iter().sum();
        (sum / self.f_len()).sqrt()
    }

    /// # Maximum Value.
    pub(crate) fn max(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.set[self.len() - 1]
        }
    }

    /// # Mean.
    pub(crate) fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else if self.unique == 1 {
            self.set[0]
        } else {
            self.total / self.f_len()
        }
    }

    /// # Minimum Value.
    pub(crate) fn min(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.set[0]
        }
    }
}

impl Abacus {
    /// # Prune Outliers.
    ///
    /// This calculates an IQR using the 5th and 95th quantiles (fuzzily), and
    /// removes entries below the lower boundary or above the upper one, using
    /// a multiplier of `1.5`.
    pub(crate) fn prune_outliers(&mut self) {
        if 1 < self.unique && 0.0 < self.deviation() {
            let q1 = self.ideal_quantile(0.05);
            let q3 = self.ideal_quantile(0.95);
            let iqr = q3 - q1;

            // Low and high boundaries.
            let lo = iqr.mul_add(-1.5, q1);
            let hi = iqr.mul_add(1.5, q3);

            // Remove outliers.
            self.set.retain(|&s| total_cmp!(lo <= s) && total_cmp!(s <= hi));

            // Recalculate totals if the length changed.
            let len = self.set.len();
            if len != self.len {
                self.len = len;
                self.unique = count_unique(&self.set);
                self.total = self.set.iter().sum();
            }
        }
    }
}

impl Abacus {
    /// # Count Above.
    ///
    /// Return the total number of entries with values larger than the target.
    fn count_above(&self, num: f64) -> usize {
        self.set.iter().rev().take_while(|&&n| total_cmp!(n > num)).count()
    }

    /// # Count Below.
    ///
    /// Return the total number of entries with values lower than the target.
    fn count_below(&self, num: f64) -> usize {
        self.set.iter().take_while(|&&n| total_cmp!(n < num)).count()
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    /// # Quantile.
    ///
    /// Return the quantile at the corresponding percentage. Values are clamped
    /// to the set's minimum and maximum, but will always correspond to a value
    /// that is actually in the set.
    fn quantile(&self, phi: f64) -> f64 {
        if self.is_empty() {
            0.0
        } else if phi <= 0.0 {
            self.min()
        } else if phi >= 1.0 {
            self.max()
        } else if self.len == 1 || self.unique == 1 {
            self.set[0]
        } else {
            // Find the absolute middle of the set.
            let target = (phi * self.f_len()).round() as usize;
            if target == 0 {
                self.min()
            } else if target >= self.len - 1 {
                self.max()
            } else {
                // The number of entries below and above our starting point.
                // Since we mathed this guess, this serves as the "ideal"
                // reference distribution.
                let target_below = target;
                let target_above = self.len.saturating_sub(target + 1);

                // Start with our best-guess value.
                let mut out = self.set[target];
                let mut diff = quantile_diff(
                    self.count_below(out),
                    self.count_above(out),
                    target_below,
                    target_above,
                );

                // See if lower values get us closer.
                let mut last = self.set[target];
                while let Some(other) = self.step_down(last) {
                    let diff2 = quantile_diff(
                        self.count_below(other),
                        self.count_above(other),
                        target_below,
                        target_above,
                    );
                    if diff2 < diff {
                        last = other;
                        out = other;
                        diff = diff2;
                    } else {
                        break;
                    }
                }

                // See if higher values get us closer.
                last = self.set[target];
                while let Some(other) = self.step_up(last) {
                    let diff2 = quantile_diff(
                        self.count_below(other),
                        self.count_above(other),
                        target_below,
                        target_above,
                    );
                    if diff2 < diff {
                        last = other;
                        out = other;
                        diff = diff2;
                    } else {
                        break;
                    }
                }

                out
            }
        }
    }

    /// # Idealized Quantile.
    ///
    /// Return the quantile at the corresponding percentage. Unlike `Abacus::quantile`,
    /// the result may not actually be present in the set. (Sparse entries are
    /// smoothed out to provide an "idealized" representation of where the cut
    /// would fall if the data were better.)
    ///
    /// This was inspired by the [`quantogram`](https://crates.io/crates/quantogram) crate's `fussy_quantile`
    /// calculations, but wound up much simpler because we have only a singular
    /// use case to worry about.
    fn ideal_quantile(&self, phi: f64) -> f64 {
        if self.is_empty() {
            0.0
        } else if phi <= 0.0 {
            self.min()
        } else if phi >= 1.0 {
            self.max()
        } else if self.len == 1 || self.unique == 1 {
            self.set[0]
        } else {
            let epsilon = 1.0 / (2.0 * self.f_len());
            let quantile = self.quantile(phi);
            if quantile == 0.0 || phi <= 1.5 * epsilon || phi >= epsilon.mul_add(-1.5, 1.0) {
                quantile
            } else {
                let lo = self.quantile(phi - epsilon);
                let hi = self.quantile(phi + epsilon);

                let lo_diff = quantile - lo;
                let hi_diff = hi - quantile;

                if lo_diff >= hi_diff * 2.0 {
                    (lo + quantile) / 2.0
                } else if hi_diff >= lo_diff * 2.0 {
                    (hi + quantile) / 2.0
                } else {
                    0.0
                }
            }
        }
    }

    /// # Step Down.
    ///
    /// Return the largest entry in the set with a value lower than the target,
    /// if any.
    fn step_down(&self, num: f64) -> Option<f64> {
        let pos = self.set.iter().position(|&n| total_cmp!(n == num))?;
        if 0 < pos {
            Some(self.set[pos - 1])
        } else {
            None
        }
    }

    /// # Step Up.
    ///
    /// Return the smallest entry in the set with a value larger than the
    /// target, if any.
    fn step_up(&self, num: f64) -> Option<f64> {
        let pos = self.set.iter().rposition(|&n| total_cmp!(n == num))?;
        if pos + 1 < self.len {
            Some(self.set[pos + 1])
        } else {
            None
        }
    }
}

/// # Count Unique.
///
/// This returns the number of unique entries in a set. It isn't particularly
/// efficient, but won't run more than twice per benchmark, so should be fine.
///
/// Note: values must be pre-sorted.
fn count_unique(src: &[f64]) -> usize {
    let mut unique = src.to_vec();
    unique.dedup_by(|a, b| total_cmp!(a == b));
    unique.len()
}

/// # Distance Above and Below.
///
/// This averages the absolute distance between the below counts and above
/// counts. An ideal distribution would return `0.0`.
fn quantile_diff(below: usize, above: usize, ref_below: usize, ref_above: usize) -> f64 {
    let below = below.abs_diff(ref_below);
    let above = above.abs_diff(ref_above);

    dactyl::int_div_float(below + above, 2).unwrap_or_default()
}

/// # Safety: This is non-zero.
const DEFAULT_SAMPLES: u32 = 16 * 4096 * 4096;
const MIN_SAMPLES: u32 = 100;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug)]
/// # Benchmark.
///
/// This struct holds a single "bench" you wish to run. See the main crate
/// documentation for more information.
pub struct Bench {
    samples: u32,
    timeout: Duration,
    stats: Option<Result<Stats, BrunchError>>,
}

impl Bench {
    #[must_use]
    /// # New.
    ///
    /// Instantiate a new benchmark with a name. The name can be anything, but
    /// is intended to represent the method call itself, like `foo::bar(10)`.
    ///
    /// Note: the names should be unique across all benchmarks, as they serve
    /// as the key used when pulling "history". If you have two totally
    /// different benchmarks named the same thing, the run-to-run change
    /// reporting won't make any sense. ;)
    ///
    /// ## Examples
    ///
    /// ```no_run
    /// use brunch::Bench;
    /// use dactyl::{NiceU8, NiceU16};
    ///
    /// brunch::benches!(
    ///     Bench::new("dactyl::NiceU8::from(0)")
    ///         .run(|| NiceU8::from(0_u8)),
    /// );
    /// ```
    ///
    /// ## Panics
    ///
    /// This method will panic if the name is empty.
    pub fn new(size: usize) -> Self {
        Self {
            samples: (DEFAULT_SAMPLES / size as u32).max(256),
            timeout: DEFAULT_TIMEOUT,
            stats: None,
        }
    }

    #[must_use]
    /// # Run Benchmark!
    ///
    /// Use this method to execute a benchmark for a callback that does not
    /// require any external arguments.
    ///
    /// ## Examples
    ///
    /// ```no_run
    /// use brunch::Bench;
    /// use dactyl::NiceU8;
    ///
    /// brunch::benches!(
    ///     Bench::new("dactyl::NiceU8::from(0)")
    ///         .run(|| NiceU8::from(0_u8))
    /// );
    /// ```
    pub fn run<F, O>(mut self, mut cb: F) -> Self
    where
        F: FnMut() -> O,
    {
        let mut times: Vec<Duration> = Vec::with_capacity(self.samples as usize);
        let now = Instant::now();

        for _ in 0..self.samples {
            let now2 = Instant::now();
            let _res = black_box(cb());
            times.push(now2.elapsed());

            if self.timeout <= now.elapsed() {
                break;
            }
        }

        self.stats.replace(Stats::try_from(times));

        self
    }
}

#[derive(Debug, PartialEq)]
enum Plot {
    time,
    troughput,
    deviation,
}

fn main() {
    let mut stats = Vec::new();
    for i in 0..15 {
        let size = 1024 * (1 << i);
        println!("{i}");
        let mut encoder: Encoder =
            Encoder::new(size, false, SymbolSize::G2x8, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder =
            Encoder::new(size / 2, false, SymbolSize::G2x16, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder =
            Encoder::new(size / 4, false, SymbolSize::G2x32, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder =
            Encoder::new(size / 8, false, SymbolSize::G2x64, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder =
            Encoder::new(size / 4, true, SymbolSize::G2x32, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder =
            Encoder::new(size / 8, true, SymbolSize::G2x64, Algorithm::ReedSalomon);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        // let mut encoder: Encoder = Encoder::new(size / 4, true, SymbolSize::G2x32, Algorithm::Sha1);
        // encoder.generate();
        // let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        // res.push(bench.stats.unwrap().unwrap());
        let mut encoder: Encoder = Encoder::new(size / 8, true, SymbolSize::G2x64, Algorithm::Sha2);
        encoder.generate();
        let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        stats.push(bench.stats.unwrap().unwrap());
        // let mut encoder: Encoder =
        //     Encoder::new(size / 4, true, SymbolSize::G2x32, Algorithm::NoCode);
        // encoder.generate();
        // let bench = Bench::new(size).run(move || black_box(encoder.encode(1)));
        // res.push(bench.stats.unwrap().unwrap());
    }
    let colors = [
        "#006400", "#00008b", "#b03060", "#ff0000", "#9467bd", "#deb887", "#00ff00", "#00ffff",
        "#ff00ff", "#6495ed",
    ];
    let encoders =
        ["RS 8", "RS 16", "RS 32", "RS 64", "RS 32 optimized", "RS 64 optimized", "Sha224"];

    println!("{stats:?}");
    for plot in [Plot::time, Plot::troughput, Plot::deviation] {
        let mut res = r#"\begin{tikzpicture}
\pgfplotsset{every axis/.append style={ultra thick},compat=1.5}
"#
        .to_string();
        for (i, color) in colors.into_iter().enumerate() {
            let [r, g, b] = from_hex(color.as_bytes()).unwrap();
            res += &format!("\\definecolor{{mycolor{i}}}{{rgb}}{{{r:.5}, {g:.5}, {b:.5}}}\n");
        }
        match plot {
            Plot::time => {
                res += r#"
    \begin{axis}[
        xlabel={message size $\quad\left(KiB\right)$},
        ylabel={wall time $\quad\left(\si{\micro\second}\right)$},
        xmajorgrids,
        enlargelimits=false,
        scaled ticks=true,
        ymajorgrids,
        width=0.99*\textwidth,
        log basis x=2,
        % ymin=0,
        % ymax=0,
        % restrict y to domain=0:1,
        xmode=log,
        ymode=log,
        x tick style={color=black},
        y tick style={color=black},
        x grid style={white!69.01960784313725!black},
        y grid style={white!69.01960784313725!black},
        legend style={at={(0.02,0.98)}, anchor=north west,legend cell align=left, align=left},
    ]
"#;
            }
            Plot::troughput => {
                res += r#"    \begin{axis}[
        xlabel={message size $\quad\left(KiB\right)$},
        ylabel={troughput $\quad\left(GiB/s\right)$},
        width=0.99*\textwidth,
        xmajorgrids,
        enlargelimits=false,
        scaled ticks=true,
        ymajorgrids,
        log basis x=2,
        xmode=log,
        x tick style={color=black},
        y tick style={color=black},
        x grid style={white!69.01960784313725!black},
        y grid style={white!69.01960784313725!black},
]
"#;
            }
            Plot::deviation => {
                res += r#"    \begin{axis}[
        xlabel={message size $\quad\left(KiB\right)$},
        ylabel={}$standard deviation\quad\quad\left(\si{\micro\second}\right)$},
        width=0.99*\textwidth,
        xmajorgrids,
        enlargelimits=false,
        scaled ticks=true,
        ymajorgrids,
        log basis x=2,
        xmode=log,
        ymode=log,
        x tick style={color=black},
        y tick style={color=black},
        x grid style={white!69.01960784313725!black},
        y grid style={white!69.01960784313725!black},
        legend style={at={(0.02,0.98)}, anchor=north west,legend cell align=left, align=left},
]
"#;
            }
        }
        for (i, encoder) in encoders.into_iter().enumerate() {
            res += &format!(
                "\\addplot[
        smooth,
        mark=diamond,
        color=mycolor{i},
    ] plot coordinates {{
    ",
            );
            for (i, stats) in stats[i..].iter().cloned().step_by(encoders.len()).enumerate() {
                let size = (1 << i) as f64;
                match plot {
                    Plot::time => res += &format!("        ({size},{:.5})\n", stats.mean * 1e6),
                    Plot::troughput => {
                        res += &format!(
                            "        ({size},{:.5})\n",
                            2.0 * size / stats.mean / (1024 * 1024) as f64
                        )
                    }
                    Plot::deviation => {
                        res += &format!(
                            "        ({size},{:.5})\n",
                            100.0 * stats.deviation / stats.mean
                        )
                    }
                }
            }
            if plot != Plot::troughput {
                res += &format!(
                    "
    }};
    \\addlegendentry{{{encoder}}}
    "
                );
            }
        }
        res += r#" 
    \end{axis}
    \end{tikzpicture}       
    "#;
        println!("{res}");
        std::fs::write(&format!("{plot:?}.tex"), &res).unwrap();
    }
}

const HASH: u8 = b'#';

pub(crate) fn from_hex(s: &[u8]) -> Result<[f64; 3], ()> {
    let mut buff: [u8; 6] = [0; 6];
    let mut buff_len = 0;

    for b in s {
        if !b.is_ascii() || buff_len == 6 {
            return Err(());
        }

        let bl = b.to_ascii_lowercase();
        if bl == HASH {
            continue;
        }
        if bl.is_ascii_hexdigit() {
            buff[buff_len] = bl;
            buff_len += 1;
        } else {
            return Err(());
        }
    }

    if buff_len == 3 {
        buff = [buff[0], buff[0], buff[1], buff[1], buff[2], buff[2]];
    }

    let hex_str = core::str::from_utf8(&buff).map_err(|_| ())?;
    let hex_digit = u32::from_str_radix(hex_str, 16).map_err(|_| ())?;

    Ok(hex_digit_to_rgb(hex_digit))
}

fn hex_digit_to_rgb(num: u32) -> [f64; 3] {
    let r = num >> 16;
    let g = (num >> 8) & 0x00FF;
    let b = num & 0x0000_00FF;

    [r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0]
}
