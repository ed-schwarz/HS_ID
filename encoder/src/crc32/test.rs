use g2p::G2Poly;
use proptest::prelude::*;

use crate::crc32::{G2x32, GENERATOR, ONE, P_X, U_PRIME, ZERO};
use crate::Field;

proptest! {

 #[test]
  fn mul_identity(a: u32, b: u32) {
    let x = GENERATOR.pow(a as usize);
    let y = GENERATOR.pow(b as usize);
    prop_assert_eq!(x.mul(y), y.mul(x));
    prop_assert_eq!(x.mul(y.add(y)), y.mul(x).add(y.mul(x)));
    prop_assert_eq!(x.mul(y.add(y)), y.mul_(x).add(y.mul_(x)).reduce());
    prop_assert_eq!(x.mul(ONE),x);
    prop_assert_eq!(x.mul(ZERO),ZERO);
  }


  #[test]
  fn mul_acceleration(a: u32, b: u32) {
        let software_res =( G2Poly(a as u64) * G2Poly(b as u64))%G2Poly(P_X as u64);
        let accelerated_res = G2x32(a).mul(G2x32(b));
        prop_assert_eq!(accelerated_res.0, software_res.0 as u32);
  }

  #[test]
  fn is_cylical(a in 1u32..u32::MAX) {
    let x = GENERATOR.pow(a as usize);
    prop_assert_eq!(x.pow((u32::MAX) as usize), ONE);
    prop_assert_eq!(G2x32(a).pow((u32::MAX) as usize), ONE);
  }

}

#[test]
fn polynom_correct() {
    let modulus = G2Poly(P_X as _);
    assert!(modulus.is_irreducible());
    assert_eq!((G2Poly((P_X as u64) << 32) / modulus).0, U_PRIME as u64 as u32 as u64);
    let generator = G2Poly(GENERATOR.0 as u64);
    assert_eq!(generator.pow_mod(u32::MAX as u64, modulus), G2Poly::UNIT);
    assert_eq!(G2Poly(91283).pow_mod(u32::MAX as u64, modulus), G2Poly::UNIT)
}

// fn print_poly(mut val: u32) {
//     let mut res = String::new();
//     for i in (0..32).rev() {
//         if (val & (1 << 31)) != 0 {
//             write!(&mut res, "x^{i} + ").unwrap()
//         }
//         val <<= 1;
//     }
//     println!("{res}")
// }
#[test]
/// some easy to understand hardcoded testcases
fn mul_acceleration_() {
    // trivial case x*y mod p == x*y
    let software_res = (G2Poly(0b100) * G2Poly(0b100)) % G2Poly(P_X as u64);
    let accelerated_res = G2x32(0b100).mul(G2x32(0b100));
    assert_eq!(accelerated_res.0, software_res.0 as u32);
    // non-trivial case x*y mod p != x*y
    let software_res = (G2Poly(GENERATOR.0 as _) * G2Poly(0b10)) % G2Poly(P_X as u64);
    let accelerated_res = GENERATOR.mul(G2x32(0b10));
    assert_eq!(accelerated_res.0, software_res.0 as u32);
}
