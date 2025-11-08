#![allow(dead_code)]

use crate::chapter2::Sum;
use crate::chapter4::Relation;
use std::collections::HashSet;
use std::hash::Hash;

pub fn relational_pair<A, B, S>(
    left: &Relation<A, S>,
    right: &Relation<B, S>,
) -> Relation<(A, B), S>
where
    A: Eq + Hash + Clone,
    B: Eq + Hash + Clone,
    S: Eq + Hash + Clone,
{
    let mut result = Relation::new();
    for (lt, src1) in left.iter() {
        for (rt, src2) in right.iter() {
            if src1 == src2 {
                result.insert((lt.clone(), rt.clone()), src1.clone());
            }
        }
    }
    result
}

pub fn relational_product<A, B, C, D>(
    left: &Relation<A, B>,
    right: &Relation<C, D>,
) -> Relation<(A, C), (B, D)>
where
    A: Eq + Hash + Clone,
    B: Eq + Hash + Clone,
    C: Eq + Hash + Clone,
    D: Eq + Hash + Clone,
{
    let mut result = Relation::new();
    for (lt, ls) in left.iter() {
        for (rt, rs) in right.iter() {
            result.insert((lt.clone(), rt.clone()), (ls.clone(), rs.clone()));
        }
    }
    result
}

pub fn relational_coproduct<A, B, C, D>(
    left: &Relation<A, C>,
    right: &Relation<B, D>,
) -> Relation<Sum<A, B>, Sum<C, D>>
where
    A: Eq + Hash + Clone,
    B: Eq + Hash + Clone,
    C: Eq + Hash + Clone,
    D: Eq + Hash + Clone,
{
    let mut result = Relation::new();
    for (lt, ls) in left.iter() {
        result.insert(Sum::Inl(lt.clone()), Sum::Inl(ls.clone()));
    }
    for (rt, rs) in right.iter() {
        result.insert(Sum::Inr(rt.clone()), Sum::Inr(rs.clone()));
    }
    result
}

pub fn power_relator_image<T, S>(relation: &Relation<T, S>, subset: &HashSet<S>) -> HashSet<T>
where
    T: Eq + Hash + Clone,
    S: Eq + Hash + Clone,
{
    relation.power_image(subset)
}

pub fn subsequences<T>(input: &[T]) -> HashSet<Vec<T>>
where
    T: Clone + Eq + Hash,
{
    fn helper<T: Clone + Eq + Hash>(rest: &[T]) -> HashSet<Vec<T>> {
        if rest.is_empty() {
            return HashSet::from([Vec::new()]);
        }
        let head = rest[0].clone();
        let tail = helper(&rest[1..]);
        let mut result = tail.clone();
        for seq in tail {
            let mut with_head = Vec::with_capacity(seq.len() + 1);
            with_head.push(head.clone());
            with_head.extend(seq);
            result.insert(with_head);
        }
        result
    }

    helper(input)
}

pub fn prefix_relation<T>(list: &[T]) -> Relation<Vec<T>, Vec<T>>
where
    T: Clone + Eq + Hash,
{
    let mut relation = Relation::new();
    for end in 0..=list.len() {
        let prefix: Vec<T> = list[..end].to_vec();
        let remainder: Vec<T> = list.to_vec();
        relation.insert(prefix, remainder.clone());
    }
    relation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter4::Relation;

    #[test]
    fn relational_pair_matches_definition() {
        let r = Relation::from_pairs(vec![("A", 1), ("B", 2)]);
        let s = Relation::from_pairs(vec![('x', 1), ('y', 2)]);
        let pair = relational_pair(&r, &s);
        assert!(pair.contains(&("A", 'x'), &1));
        assert!(pair.contains(&("B", 'y'), &2));
        assert_eq!(pair.len(), 2);
    }

    #[test]
    fn relational_product_distributes_over_converse() {
        let r = Relation::from_pairs(vec![(1, 'a'), (2, 'b')]);
        let s = Relation::from_pairs(vec![(10, 'x'), (20, 'y')]);
        let lhs = relational_product(&r, &s).converse();
        let rhs = relational_product(&r.converse(), &s.converse());
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn relational_coproduct_separates_summands() {
        let r = Relation::from_pairs(vec![("left", 1)]);
        let s = Relation::from_pairs(vec![("right", 2)]);
        let sum = relational_coproduct(&r, &s);
        assert!(sum.contains(&Sum::Inl("left"), &Sum::Inl(1)));
        assert!(sum.contains(&Sum::Inr("right"), &Sum::Inr(2)));
    }

    #[test]
    fn power_relator_image_is_monotonic() {
        let relation = Relation::from_pairs(vec![("even", 2), ("odd", 1)]);
        let mut subset_small = HashSet::new();
        subset_small.insert(2);
        let mut subset_large = subset_small.clone();
        subset_large.insert(1);
        let image_small = power_relator_image(&relation, &subset_small);
        let image_large = power_relator_image(&relation, &subset_large);
        assert!(image_small.is_subset(&image_large));
    }

    #[test]
    fn subsequences_example() {
        let subs = subsequences(&[1, 2]);
        let expected: HashSet<Vec<i32>> = HashSet::from([vec![], vec![1], vec![2], vec![1, 2]]);
        assert_eq!(subs, expected);
    }

    #[test]
    fn prefix_relation_includes_all_prefixes() {
        let prefix = prefix_relation(&['a', 'b', 'c']);
        assert!(prefix.contains(&vec!['a'], &vec!['a', 'b', 'c']));
        assert!(prefix.contains(&vec![], &vec!['a', 'b', 'c']));
    }
}
