#![allow(dead_code)]

use crate::chapter4::Relation;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

pub fn digits(mut n: u64) -> Vec<u8> {
    if n == 0 {
        return vec![0];
    }
    let mut stack = Vec::new();
    while n > 0 {
        stack.push((n % 10) as u8);
        n /= 10;
    }
    stack.reverse();
    stack
}

pub fn least_fixed_point<T, F>(mut current: HashSet<T>, f: F) -> HashSet<T>
where
    T: Eq + Hash + Clone,
    F: Fn(&HashSet<T>) -> HashSet<T>,
{
    loop {
        let next = f(&current);
        if next.is_subset(&current) {
            return current;
        }
        current.extend(next);
    }
}

#[derive(Clone, Debug)]
pub enum HyloStep<S, A> {
    Leaf(A),
    Branch(A, Vec<S>),
}

pub fn hylo<S, A, B, U, F>(seed: S, unfold: &U, fold: &F) -> B
where
    S: Clone,
    A: Clone,
    U: Fn(S) -> HyloStep<S, A>,
    F: Fn(A, Vec<B>) -> B,
    B: Clone,
{
    match unfold(seed.clone()) {
        HyloStep::Leaf(label) => fold(label, Vec::new()),
        HyloStep::Branch(label, children) => {
            let results = children
                .into_iter()
                .map(|child| hylo(child, unfold, fold))
                .collect();
            fold(label, results)
        }
    }
}

pub fn fast_pow(base: i64, exp: u64) -> i64 {
    let unfold = |state: (i64, u64)| -> HyloStep<(i64, u64), u64> {
        let (b, e) = state;
        if e == 0 {
            HyloStep::Leaf(0)
        } else if e == 1 {
            HyloStep::Leaf(1)
        } else if e % 2 == 0 {
            HyloStep::Branch(2, vec![(b, e / 2)])
        } else {
            HyloStep::Branch(3, vec![(b, e - 1)])
        }
    };
    let fold = |label: u64, mut children: Vec<i64>| -> i64 {
        match label {
            0 => 1,
            1 => base,
            2 => {
                let x = children.pop().unwrap();
                x * x
            }
            3 => {
                let x = children.pop().unwrap();
                x * base
            }
            _ => unreachable!(),
        }
    };
    hylo((base, exp), &unfold, &fold)
}

pub fn transitive_closure<T>(relation: &Relation<T, T>) -> Relation<T, T>
where
    T: Eq + Hash + Clone,
{
    let mut adjacency: HashMap<T, HashSet<T>> = HashMap::new();
    for (target, source) in relation.iter() {
        adjacency
            .entry(source.clone())
            .or_default()
            .insert(target.clone());
    }
    let mut nodes = relation.left_values();
    nodes.extend(relation.right_values());
    let mut closure = Relation::new();
    for node in &nodes {
        closure.insert(node.clone(), node.clone());
        let mut queue = VecDeque::new();
        let mut seen = HashSet::new();
        queue.push_back(node.clone());
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if seen.insert(neighbor.clone()) {
                        closure.insert(neighbor.clone(), node.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
    }
    closure
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter4::Relation;

    #[test]
    fn digits_example() {
        assert_eq!(digits(0), vec![0]);
        assert_eq!(digits(42), vec![4, 2]);
        assert_eq!(digits(1205), vec![1, 2, 0, 5]);
    }

    #[test]
    fn least_fixed_point_reaches_closure() {
        let seed = HashSet::from([0]);
        let result = least_fixed_point(seed, |set| {
            set.iter().filter(|&&n| n < 3).map(|&n| n + 1).collect()
        });
        assert_eq!(result, HashSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn fast_pow_matches_standard() {
        assert_eq!(fast_pow(2, 10), 1024);
        assert_eq!(fast_pow(3, 5), 243);
    }

    #[test]
    fn closure_adds_paths() {
        let rel = Relation::from_pairs(vec![('B', 'A'), ('C', 'B')]);
        let closure = transitive_closure(&rel);
        assert!(closure.contains(&'C', &'A'));
        assert!(closure.contains(&'B', &'A'));
        assert!(closure.contains(&'A', &'A'));
    }
}
