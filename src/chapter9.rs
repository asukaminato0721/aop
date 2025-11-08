#![allow(dead_code)]

use std::cmp::min;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditOp {
    Copy(char),
    Delete(char),
    Insert(char),
}

pub fn string_edit_distance(source: &str, target: &str) -> (usize, Vec<EditOp>) {
    let s_chars: Vec<char> = source.chars().collect();
    let t_chars: Vec<char> = target.chars().collect();
    let m = s_chars.len();
    let n = t_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    let mut back = vec![vec![EditOp::Insert(' '); n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
        if i > 0 {
            back[i][0] = EditOp::Delete(s_chars[i - 1]);
        }
    }
    for j in 0..=n {
        dp[0][j] = j;
        if j > 0 {
            back[0][j] = EditOp::Insert(t_chars[j - 1]);
        }
    }

    for i in 1..=m {
        for j in 1..=n {
            let (cost, op) = if s_chars[i - 1] == t_chars[j - 1] {
                (dp[i - 1][j - 1], EditOp::Copy(s_chars[i - 1]))
            } else {
                let delete_cost = dp[i - 1][j] + 1;
                let insert_cost = dp[i][j - 1] + 1;
                let replace_cost = dp[i - 1][j - 1] + 1;
                if delete_cost <= insert_cost && delete_cost <= replace_cost {
                    (delete_cost, EditOp::Delete(s_chars[i - 1]))
                } else if insert_cost <= replace_cost {
                    (insert_cost, EditOp::Insert(t_chars[j - 1]))
                } else {
                    (replace_cost, EditOp::Delete(s_chars[i - 1]))
                }
            };
            dp[i][j] = cost;
            back[i][j] = op;
        }
    }

    let mut ops = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        let op = back[i][j].clone();
        match op {
            EditOp::Copy(_) => {
                ops.push(op);
                i -= 1;
                j -= 1;
            }
            EditOp::Delete(_) => {
                ops.push(op);
                i -= 1;
            }
            EditOp::Insert(_) => {
                ops.push(op);
                j -= 1;
            }
        }
    }
    ops.reverse();
    (dp[m][n], ops)
}

pub fn optimal_bracketing(dimensions: &[(usize, usize)]) -> Option<usize> {
    let n = dimensions.len();
    if n == 0 {
        return None;
    }
    let mut dp = vec![vec![0usize; n]; n];
    for chain_len in 2..=n {
        for i in 0..=n - chain_len {
            let j = i + chain_len - 1;
            dp[i][j] = usize::MAX;
            for k in i..j {
                let cost =
                    dp[i][k] + dp[k + 1][j] + dimensions[i].0 * dimensions[k].1 * dimensions[j].1;
                if cost < dp[i][j] {
                    dp[i][j] = cost;
                }
            }
        }
    }
    Some(dp[0][n - 1])
}

pub fn edit_sequence_to_strings(ops: &[EditOp]) -> (String, String) {
    let mut source = String::new();
    let mut target = String::new();
    for op in ops {
        match op {
            EditOp::Copy(c) => {
                source.push(*c);
                target.push(*c);
            }
            EditOp::Delete(c) => source.push(*c),
            EditOp::Insert(c) => target.push(*c),
        }
    }
    (source, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_handles_copy_delete_insert() {
        let (dist, ops) = string_edit_distance("kitten", "sitting");
        assert_eq!(dist, 3);
        let (src, tgt) = edit_sequence_to_strings(&ops);
        assert_eq!(src, "kitten");
        assert_eq!(tgt, "sitting");
    }

    #[test]
    fn optimal_bracketing_matrix_chain() {
        let dims = vec![(10, 30), (30, 5), (5, 60)];
        assert_eq!(optimal_bracketing(&dims), Some(4500));
    }
}
