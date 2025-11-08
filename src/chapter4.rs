#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// A relation whose arrows point from the `Source` set to the `Target` set in the
/// notation of the book (`target <- source`). Internally we store the set of pairs
/// `(target, source)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation<Target, Source>
where
    Target: Eq + Hash,
    Source: Eq + Hash,
{
    pairs: HashSet<(Target, Source)>,
}

impl<Target, Source> Relation<Target, Source>
where
    Target: Eq + Hash + Clone,
    Source: Eq + Hash + Clone,
{
    pub fn iter(&self) -> impl Iterator<Item = &(Target, Source)> {
        self.pairs.iter()
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn new() -> Self {
        Self {
            pairs: HashSet::new(),
        }
    }

    pub fn from_pairs<I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (Target, Source)>,
    {
        Self {
            pairs: pairs.into_iter().collect(),
        }
    }

    pub fn insert(&mut self, target: Target, source: Source) {
        self.pairs.insert((target, source));
    }

    pub fn contains(&self, target: &Target, source: &Source) -> bool {
        self.pairs.contains(&(target.clone(), source.clone()))
    }

    pub fn subset_of(&self, other: &Self) -> bool {
        self.pairs.is_subset(&other.pairs)
    }

    pub fn meet(&self, other: &Self) -> Self {
        let mut data = HashSet::new();
        for pair in self.pairs.intersection(&other.pairs) {
            data.insert(pair.clone());
        }
        Self { pairs: data }
    }

    pub fn join(&self, other: &Self) -> Self {
        let mut data = self.pairs.clone();
        data.extend(other.pairs.iter().cloned());
        Self { pairs: data }
    }

    pub fn compose<Middle>(&self, other: &Relation<Source, Middle>) -> Relation<Target, Middle>
    where
        Middle: Eq + Hash + Clone,
    {
        let mut result = Relation::new();
        for (left, mid_left) in &self.pairs {
            for (mid_right, right) in &other.pairs {
                if mid_left == mid_right {
                    result.insert(left.clone(), right.clone());
                }
            }
        }
        result
    }

    pub fn converse(&self) -> Relation<Source, Target> {
        Relation::from_pairs(self.pairs.iter().map(|(t, s)| (s.clone(), t.clone())))
    }

    pub fn left_values(&self) -> HashSet<Target> {
        self.pairs.iter().map(|(t, _)| t.clone()).collect()
    }

    pub fn right_values(&self) -> HashSet<Source> {
        self.pairs.iter().map(|(_, s)| s.clone()).collect()
    }

    pub fn power_image(&self, subset: &HashSet<Source>) -> HashSet<Target> {
        self.pairs
            .iter()
            .filter(|(_, source)| subset.contains(source))
            .map(|(target, _)| target.clone())
            .collect()
    }

    pub fn range_coreflexive(&self) -> Relation<Target, Target> {
        Relation::<Target, Target>::identity(&self.left_values())
            .meet(&self.compose(&self.converse()))
    }

    pub fn domain_coreflexive(&self) -> Relation<Source, Source> {
        Relation::<Source, Source>::identity(&self.right_values())
            .meet(&self.converse().compose(self))
    }

    pub fn identity(elements: &HashSet<Target>) -> Relation<Target, Target> {
        Relation::from_pairs(elements.iter().map(|x| (x.clone(), x.clone())))
    }

    pub fn is_coreflexive(&self) -> bool
    where
        Target: PartialEq<Source>,
    {
        self.pairs.iter().all(|(t, s)| t == s)
    }

    pub fn is_simple(&self) -> bool {
        let mut seen: HashMap<Source, Target> = HashMap::new();
        for (t, s) in &self.pairs {
            if let Some(existing) = seen.get(s) {
                if existing != t {
                    return false;
                }
            } else {
                seen.insert(s.clone(), t.clone());
            }
        }
        true
    }

    pub fn is_entire(&self, sources: &HashSet<Source>) -> bool {
        sources.iter().all(|s| self.right_values().contains(s))
    }

    pub fn implication(
        &self,
        other: &Self,
        targets: &HashSet<Target>,
        sources: &HashSet<Source>,
    ) -> Self {
        let mut data = HashSet::new();
        for t in targets {
            for s in sources {
                if !self.contains(t, s) || other.contains(t, s) {
                    data.insert((t.clone(), s.clone()));
                }
            }
        }
        Self { pairs: data }
    }

    pub fn left_divide<NewSource>(
        &self,
        target: &Relation<Target, NewSource>,
        sources: &HashSet<Source>,
        new_sources: &HashSet<NewSource>,
    ) -> Relation<Source, NewSource>
    where
        NewSource: Eq + Hash + Clone,
    {
        let mut result = Relation::new();
        for s in sources {
            for n in new_sources {
                let mut ok = true;
                for (t, source_value) in &self.pairs {
                    if source_value == s && !target.contains(t, n) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    result.insert(s.clone(), n.clone());
                }
            }
        }
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function<Dom, Cod>
where
    Dom: Eq + Hash,
    Cod: Eq + Hash,
{
    map: HashMap<Dom, Cod>,
}

impl<Dom, Cod> Function<Dom, Cod>
where
    Dom: Eq + Hash + Clone,
    Cod: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn from_pairs<I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (Dom, Cod)>,
    {
        Self {
            map: pairs.into_iter().collect(),
        }
    }

    pub fn insert(&mut self, dom: Dom, cod: Cod) {
        self.map.insert(dom, cod);
    }

    pub fn apply(&self, dom: &Dom) -> Option<&Cod> {
        self.map.get(dom)
    }

    pub fn domain(&self) -> HashSet<Dom> {
        self.map.keys().cloned().collect()
    }

    pub fn codomain_values(&self) -> HashSet<Cod> {
        self.map.values().cloned().collect()
    }

    pub fn to_relation(&self) -> Relation<Cod, Dom> {
        Relation::from_pairs(self.map.iter().map(|(dom, cod)| (cod.clone(), dom.clone())))
    }

    pub fn after<Prev>(&self, other: &Function<Prev, Dom>) -> Function<Prev, Cod>
    where
        Prev: Eq + Hash + Clone,
    {
        let mut result = Function::new();
        for (prev, mid) in &other.map {
            if let Some(out) = self.apply(mid) {
                result.insert(prev.clone(), out.clone());
            }
        }
        result
    }

    pub fn is_monic(&self) -> bool {
        let mut seen = HashSet::new();
        for cod in self.map.values() {
            if !seen.insert(cod.clone()) {
                return false;
            }
        }
        true
    }
}

pub struct Tabulation<Target, Source>
where
    Target: Eq + Hash + Clone,
    Source: Eq + Hash + Clone,
{
    carrier: Vec<(Target, Source)>,
    left: Function<usize, Target>,
    right: Function<usize, Source>,
}

impl<Target, Source> Tabulation<Target, Source>
where
    Target: Eq + Hash + Clone,
    Source: Eq + Hash + Clone,
{
    fn new(relation: &Relation<Target, Source>) -> Self {
        let carrier: Vec<(Target, Source)> = relation.pairs.iter().cloned().collect();
        let left = Function::from_pairs(
            carrier
                .iter()
                .enumerate()
                .map(|(idx, (t, _))| (idx, t.clone())),
        );
        let right = Function::from_pairs(
            carrier
                .iter()
                .enumerate()
                .map(|(idx, (_, s))| (idx, s.clone())),
        );
        Self {
            carrier,
            left,
            right,
        }
    }

    pub fn left(&self) -> &Function<usize, Target> {
        &self.left
    }

    pub fn right(&self) -> &Function<usize, Source> {
        &self.right
    }

    pub fn relation(&self) -> Relation<Target, Source> {
        Relation::from_pairs(self.carrier.clone())
    }

    pub fn factor_through<Dom>(
        &self,
        h: &Function<Dom, Target>,
        k: &Function<Dom, Source>,
    ) -> Option<Function<Dom, usize>>
    where
        Dom: Eq + Hash + Clone,
    {
        if h.domain() != k.domain() {
            return None;
        }
        let lookup: HashMap<(Target, Source), usize> = self
            .carrier
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, pair)| (pair, idx))
            .collect();
        let mut result = Function::new();
        for dom in h.domain() {
            let left_val = h.apply(&dom)?.clone();
            let right_val = k.apply(&dom)?.clone();
            let idx = *lookup.get(&(left_val.clone(), right_val.clone()))?;
            result.insert(dom, idx);
        }
        Some(result)
    }
}

impl<Target, Source> Relation<Target, Source>
where
    Target: Eq + Hash + Clone,
    Source: Eq + Hash + Clone,
{
    pub fn tabulate(&self) -> Tabulation<Target, Source> {
        Tabulation::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set<T: Eq + Hash + Clone>(items: &[T]) -> HashSet<T> {
        items.iter().cloned().collect()
    }

    #[test]
    fn range_via_rr_converse() {
        let rel = Relation::from_pairs(vec![('a', 1), ('b', 1), ('b', 2)]);
        let range = rel.range_coreflexive();
        let diag = Relation::<char, char>::identity(&rel.left_values());
        assert_eq!(range, rel.compose(&rel.converse()).meet(&diag));
    }

    #[test]
    fn modular_law_simple_arrow() {
        let s = Relation::from_pairs(vec![(0, 'x'), (1, 'y')]); // simple
        let r = Relation::from_pairs(vec![('x', 'p'), ('y', 'q')]);
        let t = Relation::from_pairs(vec![(0, 'p'), (1, 'q')]);
        let left = s.compose(&r).meet(&t);
        let right = s.compose(&r.meet(&s.converse().compose(&t)));
        assert_eq!(left, right);
    }

    #[test]
    fn shunting_rule_for_functions() {
        let f = Relation::from_pairs(vec![(0, 'a'), (1, 'b')]);
        let r = Relation::from_pairs(vec![('a', 1), ('b', 2)]);
        let s = Relation::from_pairs(vec![(0, 1), (1, 2)]);
        assert!(f.compose(&r).subset_of(&s) == r.subset_of(&f.converse().compose(&s)));
    }

    #[test]
    fn tabulation_factorises() {
        let relation = Relation::from_pairs(vec![('x', 1), ('y', 2)]);
        let tab = relation.tabulate();
        let indices = set(&['p', 'q']);
        let m = Function::from_pairs(vec![('p', 0usize), ('q', 1usize)]);
        let h = tab.left().after(&m);
        let k = tab.right().after(&m);
        let recovered = tab.factor_through(&h, &k).expect("factorisation exists");
        assert_eq!(m.domain(), recovered.domain());
        for key in indices {
            assert_eq!(m.apply(&key), recovered.apply(&key));
        }
    }
}
