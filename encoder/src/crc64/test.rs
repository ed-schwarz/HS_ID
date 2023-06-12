use proptest::prelude::*;

use crate::crc64::{G2x64, ONE, P_X, ZERO};
use crate::Field;

proptest! {

 #[test]
  fn mul_identity(a: u64, b: u64) {
    let x = G2x64(a);
    let y = G2x64(b);
    prop_assert_eq!(x.mul(y), y.mul(x));
    prop_assert_eq!(x.mul(y.add(y)), y.mul(x).add(y.mul(x)));
    prop_assert_eq!(x.mul(y.add(y)), y.mul_(x).add(y.mul_(x)).reduce());
    prop_assert_eq!(x.mul(ONE),x);
    prop_assert_eq!(x.mul(ZERO),ZERO);
  }

  #[test]
  fn  mul_acceleration(a: u64, b: u64) {
    let x = G2x64(a);
    let y = G2x64(b);
    println!("{a:b}");
    println!("{b:b}  {}, {}", b.leading_zeros(), x.mul(y).0);
    prop_assert_eq!(x.mul(y).0, software_rem(x.mul_(y).into()));
  }

  #[test]
  fn is_cylical(a in 1u64..u64::MAX) {
    let x = G2x64(a);
    prop_assert_eq!(x.pow(u64::MAX), ONE);
  }

}

// fn print_poly(mut val: u64) {
//     let mut res = String::new();
//     for i in (0..64).rev() {
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
    println!("fooo {:x}", software_div());
    // trivial case x*y mod p == x*y
    let accelerated_res = G2x64(0b100).mul(G2x64(0b100));
    assert_eq!(0b10000, accelerated_res.0);
    let accelerated_res = G2x64(0b10).mul(G2x64(1u64 << 63));
    assert_eq!(accelerated_res.0, P_X as u64);
    let accelerated_res = G2x64(0b11).mul(G2x64(1u64 << 63));
    assert_eq!(accelerated_res.0, P_X as u64 ^ 1u64 << 63);
    let accelerated_res = G2x64(0b100).mul(G2x64(1u64 << 63));
    assert_eq!(accelerated_res.0, software_rem(G2x64(0b100).mul_(G2x64(1u64 << 63)).into()));
}

fn software_rem(val: u128) -> u64 {
    let module = P_X as u64 as u128 | (1u128 << 64);
    let mod_degree_p1 = 128 - module.leading_zeros();
    assert!(mod_degree_p1 > 0);

    let mut rem = val;
    let mut rem_degree_p1 = 128 - rem.leading_zeros();

    while mod_degree_p1 <= rem_degree_p1 {
        let shift_len = rem_degree_p1 - mod_degree_p1;
        rem ^= module << shift_len;
        rem_degree_p1 = 128 - rem.leading_zeros();
    }

    // NB: rem_degree < mod_degree implies that rem < mod so it fits in u64
    rem as u64
}
fn software_div() -> u64 {
    let module = P_X as u64 as u128 | (1u128 << 64);
    let mod_degree_p1 = 128 - module.leading_zeros();
    assert!(mod_degree_p1 > 0);

    let mut rem = (P_X as u64 as u128) << 64;
    let mut res = 0;
    let mut rem_degree_p1 = 128 - rem.leading_zeros();

    while mod_degree_p1 <= rem_degree_p1 {
        let shift_len = rem_degree_p1 - mod_degree_p1;
        res |= 1u128 << shift_len;
        rem ^= module << shift_len;
        rem_degree_p1 = 128 - rem.leading_zeros();
    }
    // NB: rem_degree < mod_degree implies that rem < mod so it fits in u64
    res as u64
}
