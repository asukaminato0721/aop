#![allow(dead_code)]

use crate::chapter1::BinaryTree;
use crate::chapter2::list;

/// Computes `(sum, length)` for a list of natural numbers using a single fold
/// (banana-split law in action).
pub fn banana_split_stats(list_values: list::List<u64>) -> (u64, usize) {
    list::fold(list_values, (0u64, 0usize), |head, (sum, len)| {
        (sum + head, len + 1)
    })
}

/// Average of a list of naturals; returns `None` for the empty list.
pub fn average(list_values: list::List<u64>) -> Option<f64> {
    let (sum, len) = banana_split_stats(list_values);
    if len == 0 {
        None
    } else {
        Some(sum as f64 / len as f64)
    }
}

/// Ruby triangle operator specialised to lists of integers using a simple numeric function.
pub fn triangle_fn(list_values: list::List<u64>, f: fn(u64) -> u64) -> list::List<u64> {
    let mut output = Vec::new();
    for (idx, mut value) in list::to_vec(list_values).into_iter().enumerate() {
        for _ in 0..idx {
            value = f(value);
        }
        output.push(value);
    }
    list::from_vec(output)
}

/// Depth of a binary tree computed with a single fold (mirrors the Horner-style optimisation).
pub fn tree_depth_fast(tree: &BinaryTree<u64>) -> u64 {
    fn succ(value: u64) -> u64 {
        value + 1
    }

    fn combine(left: u64, right: u64) -> u64 {
        succ(left.max(right))
    }

    tree.fold(&|_| 1u64, &combine)
}

/// Integer-only TeX rounding (`intern`) for decimals represented as digits.
pub fn intern_decimal(digits: &[u8]) -> u32 {
    if digits.is_empty() {
        return 0;
    }

    let mut numerator: u128 = 0;
    for &digit in digits {
        numerator = numerator * 10 + digit as u128;
    }

    let denom = 10u128.pow(digits.len() as u32);
    let scaled = numerator * 65536u128;
    let rounded = (scaled + denom / 2) / denom;
    rounded as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banana_split_matches_two_pass() {
        let list = list::from_vec(vec![1, 2, 3, 4]);
        let (sum, len) = banana_split_stats(list);
        assert_eq!(sum, 10);
        assert_eq!(len, 4);
    }

    #[test]
    fn average_handles_empty_case() {
        assert_eq!(average(list::nil()), None);
        let avg = average(list::from_vec(vec![2, 4, 6])).unwrap();
        assert!((avg - 4.0).abs() < 1e-9);
    }

    #[test]
    fn triangle_with_successor() {
        let list = list::from_vec(vec![1, 2, 3]);
        let triangle = triangle_fn(list, |x| x + 1);
        assert_eq!(list::to_vec(triangle), vec![1, 3, 5]);
    }

    #[test]
    fn tree_depth_matches_previous_definition() {
        use crate::chapter1::BinaryTree;
        let tree = BinaryTree::Bin(
            Box::new(BinaryTree::Tip(1)),
            Box::new(BinaryTree::Bin(
                Box::new(BinaryTree::Tip(2)),
                Box::new(BinaryTree::Tip(3)),
            )),
        );
        assert_eq!(tree_depth_fast(&tree), 3);
    }

    #[test]
    fn intern_decimal_matches_float_reference() {
        let digits = vec![1, 2, 5];
        let result = intern_decimal(&digits);
        let mut frac = 0.0;
        for (i, digit) in digits.iter().enumerate() {
            frac += (*digit as f64) / 10f64.powi((i as i32) + 1);
        }
        let float = ((frac * 65536.0) + 0.5).floor() as u32;
        assert_eq!(result, float);
    }
}
