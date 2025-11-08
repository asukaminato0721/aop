#![allow(dead_code)]

use crate::chapter3::intern_decimal;

pub fn tex_extern(n: u32, max_len: usize) -> Option<Vec<u8>> {
    let pow10 = |len: usize| -> u64 { 10u64.pow(len as u32) };
    for len in 1..=max_len {
        let factor = pow10(len);
        let target = n as u64 * factor / 65536u64;
        let mut candidates = Vec::new();
        for delta in -2i64..=2 {
            let cand = if delta < 0 {
                target.checked_sub((-delta) as u64)
            } else {
                target.checked_add(delta as u64)
            };
            if let Some(value) = cand {
                if value < factor {
                    candidates.push(value);
                }
            }
        }
        candidates.sort();
        candidates.dedup();
        for cand in candidates {
            let digits = to_digits(cand, len);
            if intern_decimal(&digits) == n {
                return Some(digits);
            }
        }
    }
    None
}

fn to_digits(mut value: u64, len: usize) -> Vec<u8> {
    let mut digits = vec![0u8; len];
    for digit in digits.iter_mut().rev() {
        *digit = (value % 10) as u8;
        value /= 10;
    }
    digits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tex_extern_roundtrips_small_values() {
        for &value in &[0u32, 1, 1234, 65535] {
            let digits = tex_extern(value, 6).expect("representation");
            assert_eq!(crate::chapter3::intern_decimal(&digits), value);
        }
    }

    #[test]
    fn tex_extern_prefers_short_representations() {
        let digits = tex_extern(32768, 4).unwrap();
        assert!(digits.len() <= 4);
    }
}
