use std::arch::x86_64 as arch;

#[cfg(test)]
mod test;

const K6: i64 = 0x490D678D;
const K5: i64 = 0xF200AA66;
const P_X: i64 = 0x104C11DB7;
const U_PRIME: i64 = 0x104D101DF;

const GENERATOR: G2x32 = G2x32(1u32 << 31);
#[cfg(test)]
const ONE: G2x32 = G2x32(1);
#[cfg(test)]
const ZERO: G2x32 = G2x32(0);

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct G2x32(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct G2x32Product(arch::__m128i);

impl G2x32Product {
    pub fn reduce(self) -> G2x32 {
        unsafe { G2x32(reduce128(self.0)) }
    }

    pub fn add(self, other: Self) -> Self {
        unsafe { G2x32Product(arch::_mm_xor_si128(self.0, other.0)) }
    }
}

impl From<G2x32> for G2x32Product {
    fn from(value: G2x32) -> Self {
        unsafe { Self(arch::_mm_set_epi32(0, 0, 0, value.0 as i32)) }
    }
}

impl crate::Field for G2x32 {
    type Data = u32;

    const N: usize = u32::MAX as usize;

    const GENERATOR: Self = GENERATOR;

    fn pow(self, p: usize) -> Self {
        let mut val = 1;
        let mut pow_pos = 1 << (::std::mem::size_of::<u32>() * 8 - 1);
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
        crate::encode_rs32(data, cue)
    }
}

impl G2x32 {
    pub fn mul_(self, other: Self) -> G2x32Product {
        let res = unsafe { mul(self.0, other.0) };
        G2x32Product(res)
    }

    pub fn mul2(self, other1: Self, other2: Self) -> G2x32Product {
        unsafe {
            let arg = arch::_mm_set_epi32(0, self.0 as i32, 0, other1.0 as i32);
            let res = arch::_mm_clmulepi64_si128(arg, arg, 0x10);
            let arg = arch::_mm_insert_epi32(res, other2.0 as i32, 2);
            let res = arch::_mm_clmulepi64_si128(arg, arg, 0x10);
            G2x32Product(res)
        }
    }
}

impl From<u32> for G2x32 {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

unsafe fn mul(lhs: u32, rhs: u32) -> arch::__m128i {
    let arg = arch::_mm_set_epi32(0, rhs as i32, 0, lhs as i32);
    arch::_mm_clmulepi64_si128(arg, arg, 0x10)
}

unsafe fn squared(x: u32) -> u32 {
    let arg = arch::_mm_set_epi32(0, 0, 0, x as i32);
    let product = arch::_mm_clmulepi64_si128(arg, arg, 0x00);
    // let squared = arch::_mm_clmulepi64_si128(arg, arg, 0x00);
    baret_reduce(product)
}

/// reduces a 128 bit polynom to 32 bits using the following equation:
///
/// M = Ax^96 + Bx^64 + C
/// M mod P = A (x^96 mod P) + B (x^64 mod P) + C
///
/// where M is the 128bit message, P is the crc32 polynom
/// A and B are the MS 32bit integers and C is the lest significant 64 bit integer
unsafe fn reduce128(x: arch::__m128i) -> u32 {
    let constants = arch::_mm_set_epi64x(K5, K6);
    let h = arch::_mm_clmulepi64_si128(arch::_mm_srli_si128(x, 4), constants, 0x11);
    let l = arch::_mm_clmulepi64_si128(
        arch::_mm_and_si128(x, arch::_mm_set_epi32(0, !0, 0, 0)),
        constants,
        0x01,
    );
    let x = arch::_mm_xor_si128(arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, !0, !0)), h);
    let x = arch::_mm_xor_si128(x, l);
    baret_reduce(x)
}

/// Perform a Barrett reduction from our now 64 bits to 32 bits as described in
/// Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction
unsafe fn baret_reduce(x: arch::__m128i) -> u32 {
    let pu = arch::_mm_set_epi64x(U_PRIME, P_X);
    // T1(x) = ⌊(R(x) / x^32)⌋ • μ
    let t1 = arch::_mm_clmulepi64_si128(
        arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, !0, 0)),
        pu,
        0x10,
    );
    // T2(x) = ⌊(T1(x) / x^32)⌋ • P(x)
    let t2 = arch::_mm_clmulepi64_si128(
        arch::_mm_and_si128(t1, arch::_mm_set_epi32(0, !0, 0, 0)),
        pu,
        0x01,
    );

    // C(x) = R(x) ^ T2(x) / x^32
    let tmp = arch::_mm_extract_epi32(t2, 0) as u32;
    let x = arch::_mm_extract_epi32(x, 0) as u32;

    tmp ^ x
}
