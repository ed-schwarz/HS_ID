use proptest::prelude::*;

use crate::{encode_rs, encode_rs32, G2x32};

// #[test]
// fn smoke_test() {
//     let encoder = super::Encoder::new(, )
// }
proptest! {

   #[test]
    fn mul_identity(data: Vec<u32>, cue: u32) {
        let slow_result = encode_rs::<G2x32>(&data, cue as usize);
        let fast_result = encode_rs32(&data, cue as usize);
        prop_assert_eq!(slow_result,fast_result);
    }

}
