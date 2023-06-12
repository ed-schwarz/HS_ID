use std::arch::x86_64::{self as arch};
use std::mem::transmute;

#[cfg(test)]
mod test;

const P_X: i64 = 0xad93d23594c935a9u64 as i64;
const U_PRIME: i64 = 0xddf3eeb298be6cf8u64 as i64;

#[cfg(test)]
const ONE: G2x64 = G2x64(1);
#[cfg(test)]
const ZERO: G2x64 = G2x64(0);

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct G2x64(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct G2x64Product(arch::__m128i);

impl G2x64Product {
    pub fn reduce(self) -> G2x64 {
        unsafe { G2x64(baret_reduce(self.0)) }
    }

    pub fn add(self, other: Self) -> Self {
        unsafe { G2x64Product(arch::_mm_xor_si128(self.0, other.0)) }
    }
}

impl From<G2x64> for G2x64Product {
    fn from(value: G2x64) -> Self {
        unsafe { Self(arch::_mm_set_epi64x(0, value.0 as i64)) }
    }
}
impl From<G2x64Product> for u128 {
    fn from(value: G2x64Product) -> Self {
        unsafe { transmute(value.0) }
    }
}

impl crate::Field for G2x64 {
    type Data = u64;

    const N: usize = u64::MAX as usize;

    fn mul(self, other: Self) -> Self {
        unsafe {
            let res = mul(self.0, other.0);
            Self(baret_reduce(res))
        }
    }

    fn add(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    fn encode_rs(data: &[Self::Data], cue: usize) -> Self {
        crate::encode_rs64(data, cue)
    }
}

impl G2x64 {
    pub fn mul_(self, other: Self) -> G2x64Product {
        let res = unsafe { mul(self.0, other.0) };
        G2x64Product(res)
    }
    pub fn pow(self, p: u64) -> Self {
        let mut val = 1;
        let mut pow_pos = 1 << 63;
        // assert_eq!(pow_pos << 1, 0);
        while pow_pos > 0 {
            if val != 1 {
                val = unsafe { squared(val) };
                if (pow_pos & p) > 0 {
                    val = unsafe { baret_reduce(mul(val, self.0)) };
                }
            } else if (pow_pos & p) > 0 {
                val = self.0;
            }
            pow_pos >>= 1;
        }
        Self(val)
    }
}

impl From<u64> for G2x64 {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

unsafe fn mul(lhs: u64, rhs: u64) -> arch::__m128i {
    let arg = arch::_mm_set_epi64x(rhs as i64, lhs as i64);
    arch::_mm_clmulepi64_si128(arg, arg, 0x10)
}

unsafe fn squared(x: u64) -> u64 {
    let arg = arch::_mm_set_epi64x(0, x as i64);
    let product = arch::_mm_clmulepi64_si128(arg, arg, 0x00);
    // let squared = arch::_mm_clmulepi64_si128(arg, arg, 0x00);
    baret_reduce(product)
}

/// Perform a Barrett reduction from our 128 bits to 64 bits as described in
/// Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction
unsafe fn baret_reduce(x: arch::__m128i) -> u64 {
    let pu = arch::_mm_set_epi64x(U_PRIME, P_X);
    let mut t1 = arch::_mm_clmulepi64_si128(x, pu, 0x11);
    t1 = arch::_mm_xor_si128(t1, x);
    let t2 = arch::_mm_clmulepi64_si128(t1, pu, 0x01);
    let reduced = arch::_mm_xor_si128(t2, x);
    arch::_mm_extract_epi64(reduced, 0) as u64
}
