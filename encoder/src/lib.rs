use std::mem::size_of;
use std::slice;

use num_traits::identities::zero;
use num_traits::PrimInt;
use pyo3::prelude::*;
use rand::{thread_rng, Fill};
use sha1::{Digest, Sha1};

mod crc32;
mod crc64;

#[cfg(test)]
mod test;

pub use crc32::G2x32;

use crate::crc32::G2x32Product;
use crate::crc64::{G2x64, G2x64Product};

/// A Python module implemented in Rust.
#[pymodule]
fn encoder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Algorithm>()?;
    m.add_class::<SymbolSize>()?;
    m.add_class::<Encoder>()?;
    Ok(())
}

#[pyclass]
#[derive(Clone, Copy)]
pub enum Algorithm {
    ReedSalomon,
    NoCode,
    Sha1,
    Sha2,
}

#[pyclass]
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum SymbolSize {
    G2x8 = 1,
    G2x16 = 2,
    G2x32 = 4,
    G2x64 = 8,
}
type Word = u64;
const WORD_SIZE: usize = size_of::<Word>();

#[pyclass]
pub struct Encoder {
    algorithm: Algorithm,
    symbol_size: SymbolSize,
    specialize: bool,
    num_symbols: usize,
    data1: Box<[Word]>,
    data2: Box<[Word]>,
}

#[pymethods]
impl Encoder {
    #[new]
    pub fn init(k: u32, symbol_size: SymbolSize, algorithm: Algorithm) -> Self {
        Self::new(k as usize, true, symbol_size, algorithm)
    }

    pub fn generate(&mut self) {
        self.data1.try_fill(&mut rand::thread_rng()).unwrap();
        loop {
            self.data2.try_fill(&mut rand::thread_rng()).unwrap();
            if self.data1 != self.data2 {
                break;
            }
        }
    }

    pub fn encode(&self, tags: usize) -> usize {
        match self.symbol_size {
            SymbolSize::G2x8 => unsafe { self.encode_inner::<G2x8>(tags) },
            SymbolSize::G2x16 => unsafe { self.encode_inner::<G2x16>(tags) },
            SymbolSize::G2x32 => unsafe { self.encode_inner::<G2x32>(tags) },
            SymbolSize::G2x64 => unsafe { self.encode_inner::<G2x64>(tags) },
        }
    }
}

impl Encoder {
    pub fn new(
        num_symbols: usize,
        specialize: bool,
        symbol_size: SymbolSize,
        algorithm: Algorithm,
    ) -> Self {
        let data_size = num_symbols * symbol_size as usize;
        Self {
            algorithm,
            symbol_size,
            specialize,
            num_symbols,
            data1: vec![0; data_size / WORD_SIZE + (data_size % WORD_SIZE != 0) as usize]
                .into_boxed_slice(),
            data2: vec![0; data_size / WORD_SIZE + (data_size % WORD_SIZE != 0) as usize]
                .into_boxed_slice(),
        }
    }
    unsafe fn encode_inner<T: Field>(&self, tags: usize) -> usize {
        let num_tags = match self.algorithm {
            Algorithm::ReedSalomon { .. } => (1usize << (self.symbol_size as usize * 8 - 1)) - 1,
            Algorithm::NoCode => self.num_symbols,
            Algorithm::Sha1 => 5 * 4 / self.symbol_size as usize,
            Algorithm::Sha2 => 7 * 4 / self.symbol_size as usize,
        };
        rand::seq::index::sample(&mut thread_rng(), num_tags, tags.min(num_tags))
            .iter()
            .filter(|&tag| self.encode_single::<T>(tag))
            .count()
    }

    unsafe fn encode_single<T: Field>(&self, cue: usize) -> bool {
        match self.algorithm {
            Algorithm::ReedSalomon if self.specialize => {
                let id1 = T::encode_rs(self.data1_symbols(), cue);
                let id2 = T::encode_rs(self.data2_symbols(), cue % T::N);
                id1 == id2
            }
            Algorithm::ReedSalomon => {
                let id1 = encode_rs::<T>(self.data1_symbols(), cue);
                let id2 = encode_rs::<T>(self.data2_symbols(), cue);
                id1 == id2
            }
            Algorithm::NoCode => self.data1_symbols::<T>()[cue] == self.data2_symbols::<T>()[cue],
            Algorithm::Sha1 => {
                let id1 = Sha1::digest(self.data1_bytes());
                let id2 = Sha1::digest(self.data2_bytes());
                id1.as_slice()[..size_of::<T::Data>()] == id2.as_slice()[..size_of::<T::Data>()]
            }
            Algorithm::Sha2 => {
                let id1 = sha2::Sha224::digest(self.data1_bytes());
                let id2 = sha2::Sha224::digest(self.data2_bytes());
                id1.as_slice()[..size_of::<T::Data>()] == id2.as_slice()[..size_of::<T::Data>()]
            }
        }
    }

    fn data1_bytes(&self) -> &[u8] {
        let ptr = self.data1.as_ptr() as *const u8;
        let len = self.symbol_size as usize * self.num_symbols;
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    fn data2_bytes(&self) -> &[u8] {
        let ptr = self.data2.as_ptr() as *const u8;
        let len = self.symbol_size as usize * self.num_symbols;
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    unsafe fn data1_symbols<T>(&self) -> &[T] {
        let ptr = self.data1.as_ptr() as *const T;
        unsafe { slice::from_raw_parts(ptr, self.num_symbols) }
    }

    unsafe fn data2_symbols<T>(&self) -> &[T] {
        let ptr = self.data2.as_ptr() as *const T;
        unsafe { slice::from_raw_parts(ptr, self.num_symbols) }
    }
}

g2p::g2p!(G2x8, 8);

impl Field for G2x8 {
    type Data = u8;
    const N: usize = u8::MAX as usize;

    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
}

g2p::g2p!(G2x16, 16);

impl Field for G2x16 {
    type Data = u16;
    const N: usize = u16::MAX as usize;

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn add(self, other: Self) -> Self {
        self + other
    }
}

pub trait Field: From<Self::Data> + PartialEq + Copy {
    type Data: PrimInt + TryFrom<usize>;
    const N: usize;
    fn mul(self, other: Self) -> Self;
    fn add(self, other: Self) -> Self;
    fn encode_rs(data: &[Self::Data], cue: usize) -> Self {
        encode_rs(data, cue)
    }
}

pub fn encode_rs<T: Field>(data: &[T::Data], cue: usize) -> T {
    if data.is_empty() {
        return zero::<T::Data>().into();
    }
    let mut res: T = data[0].into();
    let initial_generator: T = T::Data::try_from(cue).map_err(|_| cue).unwrap().into();
    let mut generator: T = initial_generator;
    for symbol in data[1..].iter().copied() {
        if symbol != zero() {
            let symbol: T = symbol.into();
            res = res.add(generator.mul(symbol));
        }
        generator = generator.mul(initial_generator);
    }
    res
}

pub fn encode_rs32(data: &[u32], cue: usize) -> G2x32 {
    if data.is_empty() {
        return G2x32(0);
    }
    let mut res: G2x32Product = G2x32(data[0]).into();
    let initial_generator = G2x32(cue as u32);
    let mut generator = initial_generator;
    let mut lut =
        vec![generator; ((data.len().saturating_sub(1) as f64).sqrt() as usize).min(4096)];
    if !lut.is_empty() {
        for entry in &mut lut[1..] {
            *entry = generator.mul(initial_generator);
            generator = *entry;
        }
        res = res.add(G2x32(data[1]).mul_(initial_generator));
        generator = initial_generator;
        for chunk in data[2..].chunks(lut.len()) {
            let mut chunk_sum: G2x32Product = G2x32(0).into();
            for (&sym, &generator_pow) in chunk.iter().zip(&lut) {
                chunk_sum = chunk_sum.add(G2x32(sym).mul_(generator_pow))
            }
            res = res.add(chunk_sum.mul(generator.into()));
            generator = generator.mul(*lut.last().unwrap());
        }
    }
    res.reduce()
}

pub fn encode_rs64(data: &[u64], cue: usize) -> G2x64 {
    if data.is_empty() {
        return G2x64(0);
    }
    let mut res: G2x64Product = G2x64(data[0]).into();
    let initial_generator = G2x64(cue as u64);
    let mut generator = initial_generator;
    let mut lut =
        vec![generator; ((data.len().saturating_sub(1) as f64).sqrt() as usize).min(4096)];
    if !lut.is_empty() {
        for entry in &mut lut[1..] {
            *entry = generator.mul(initial_generator);
            generator = *entry;
        }
        res = res.add(G2x64(data[1]).mul_(initial_generator));
        generator = initial_generator;
        for chunk in data[2..].chunks(lut.len()) {
            let mut chunk_sum: G2x64Product = G2x64(0).into();
            for (&sym, &generator_pow) in chunk.iter().zip(&lut) {
                chunk_sum = chunk_sum.add(G2x64(sym).mul_(generator_pow))
            }
            res = res.add(chunk_sum.reduce().mul_(generator));
            generator = generator.mul(*lut.last().unwrap());
        }
    }
    res.reduce()
}
