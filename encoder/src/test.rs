use proptest::prelude::*;

use crate::crc64::G2x64;
use crate::{encode_rs, encode_rs32, encode_rs64, G2x32};

// #[test]
// fn smoke_test() {
//     let encoder = super::Encoder::new(, )
// }
proptest! {

   #[test]
    fn fast_eq_slow32(data: Vec<u32>, cue: u32) {
        let slow_result = encode_rs::<G2x32>(&data, cue as usize);
        let fast_result = encode_rs32(&data, cue as usize);
        prop_assert_eq!(slow_result,fast_result);
    }
    fn fast_eq_slow64(data: Vec<u64>, cue: u64) {
        let slow_result = encode_rs::<G2x64>(&data, cue as usize);
        let fast_result = encode_rs64(&data, cue as usize);
        prop_assert_eq!(slow_result,fast_result);
    }

}
