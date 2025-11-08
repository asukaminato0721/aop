#![allow(dead_code)]

use crate::chapter4::Relation;
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::hash::Hash;

pub fn thin_set<T, R>(set: &HashSet<T>, rel: R) -> HashSet<T>
where
    T: Eq + Hash + Clone,
    R: Fn(&T, &T) -> bool,
{
    set.iter()
        .filter(|candidate| {
            for opponent in set {
                if rel(opponent, candidate) && !rel(candidate, opponent) {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect()
}

pub fn thinlist<T, R>(list: &[T], rel: R) -> Vec<T>
where
    T: Clone,
    R: Fn(&T, &T) -> bool,
{
    let mut kept_rev: Vec<T> = Vec::new();
    'outer: for item in list.iter().rev() {
        for candidate in &kept_rev {
            if rel(candidate, item) && !rel(item, candidate) {
                continue 'outer;
            }
        }
        kept_rev.push(item.clone());
    }
    kept_rev.reverse();
    kept_rev
}

pub fn thin_network_paths<T>(graph: &Relation<T, T>, sources: &[T]) -> Relation<Vec<T>, T>
where
    T: Eq + Hash + Clone + Ord,
{
    let mut adjacency = std::collections::HashMap::<T, Vec<T>>::new();
    for (target, source) in graph.iter() {
        adjacency
            .entry(source.clone())
            .or_default()
            .push(target.clone());
    }

    let mut frontier: VecDeque<(Vec<T>, T)> = VecDeque::new();
    for source in sources {
        frontier.push_back((vec![source.clone()], source.clone()));
    }

    let mut best_paths: Relation<Vec<T>, T> = Relation::new();
    let mut visited = std::collections::HashMap::<T, BTreeSet<Vec<T>>>::new();

    while let Some((path, node)) = frontier.pop_front() {
        best_paths.insert(path.clone(), node.clone());
        if let Some(edges) = adjacency.get(&node) {
            for next in edges {
                let mut candidate = path.clone();
                candidate.push(next.clone());
                let entry = visited.entry(next.clone()).or_default();
                if entry.insert(candidate.clone()) {
                    frontier.push_back((candidate, next.clone()));
                }
            }
        }
    }

    best_paths
}

pub fn layered_shortest_path<T, F>(layers: &[Vec<T>], weight: F) -> Option<(i64, Vec<T>)>
where
    T: Eq + Hash + Clone,
    F: Fn(&T, &T) -> Option<i64>,
{
    if layers.len() < 2 {
        return None;
    }
    use std::collections::HashMap;
    let mut next_map: HashMap<T, (i64, Vec<T>)> = HashMap::new();
    for node in layers.last()? {
        next_map.insert(node.clone(), (0, vec![node.clone()]));
    }
    for layer in layers[..layers.len() - 1].iter().rev() {
        let mut current: HashMap<T, (i64, Vec<T>)> = HashMap::new();
        for node in layer {
            let mut best: Option<(i64, Vec<T>)> = None;
            for (next, (cost_next, path_next)) in &next_map {
                if let Some(edge) = weight(node, next) {
                    let total = edge + cost_next;
                    let mut candidate_path = vec![node.clone()];
                    candidate_path.extend(path_next.clone());
                    match &best {
                        None => best = Some((total, candidate_path)),
                        Some((best_cost, _)) if total < *best_cost => {
                            best = Some((total, candidate_path))
                        }
                        _ => {}
                    }
                }
            }
            if let Some(entry) = best {
                current.insert(node.clone(), entry);
            } else {
                return None;
            }
        }
        next_map = current;
    }
    let mut final_best: Option<(i64, Vec<T>)> = None;
    for node in layers.first()? {
        if let Some(result) = next_map.get(node) {
            match &final_best {
                None => final_best = Some(result.clone()),
                Some((best_cost, _)) if result.0 < *best_cost => final_best = Some(result.clone()),
                _ => {}
            }
        }
    }
    final_best
}

pub fn bitonic_tour_cost(points: &[(f64, f64)]) -> Option<f64> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut indexed: Vec<(usize, (f64, f64))> = points.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());
    let mut dp = vec![vec![f64::INFINITY; n]; n];

    let dist = |i: usize, j: usize| -> f64 {
        let dx = indexed[i].1.0 - indexed[j].1.0;
        let dy = indexed[i].1.1 - indexed[j].1.1;
        (dx * dx + dy * dy).sqrt()
    };

    dp[0][1] = dist(0, 1);
    for j in 2..n {
        for i in 0..j - 1 {
            dp[i][j] = dp[i][j - 1] + dist(j - 1, j);
        }
        let mut best = f64::INFINITY;
        for k in 0..j - 1 {
            let candidate = dp[k][j - 1] + dist(k, j);
            if candidate < best {
                best = candidate;
            }
        }
        dp[j - 1][j] = best;
    }

    let mut answer = f64::INFINITY;
    for k in 0..n - 1 {
        let candidate = dp[k][n - 1] + dist(k, n - 1);
        if candidate < answer {
            answer = candidate;
        }
    }
    Some(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_set_removes_dominated_elements() {
        let set: HashSet<_> = [1, 2, 3].into_iter().collect();
        let thinned = thin_set(&set, |a, b| a <= b);
        assert_eq!(thinned, HashSet::from([1]));
    }

    #[test]
    fn thinlist_preserves_order() {
        let list = vec![5, 3, 4, 2, 1];
        let thinned = thinlist(&list, |a, b| a >= b);
        assert_eq!(thinned, vec![5, 4, 2, 1]);
    }

    #[test]
    fn thin_network_paths_finds_paths() {
        let rel = Relation::from_pairs(vec![('B', 'A'), ('C', 'B')]);
        let relation = thin_network_paths(&rel, &['A']);
        assert!(relation.contains(&vec!['A', 'B', 'C'], &'C'));
    }

    #[test]
    fn layered_shortest_path_prefers_cheapest_route() {
        let layers = vec![vec!['A', 'B'], vec!['C'], vec!['D']];
        let weight = |from: &char, to: &char| match (from, to) {
            ('A', 'C') => Some(5),
            ('B', 'C') => Some(1),
            ('C', 'D') => Some(3),
            _ => None,
        };
        let result = layered_shortest_path(&layers, weight).expect("path");
        assert_eq!(result.0, 4);
        assert_eq!(result.1, vec!['B', 'C', 'D']);
    }

    #[test]
    fn bitonic_tour_cost_on_line() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let cost = bitonic_tour_cost(&pts).unwrap();
        assert!((cost - 6.0).abs() < 1e-6);
    }
}
