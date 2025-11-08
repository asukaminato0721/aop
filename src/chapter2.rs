#![allow(dead_code)]

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

// -----------------------------------------------------------------------------
// Basic categorical combinators (terminal objects, products, coproducts)
// -----------------------------------------------------------------------------

/// The unique arrow `! : A → 1`. Discards its input and returns the terminal value.
pub fn bang<T>(_value: T) {}

/// Projection onto the first component (outl).
pub fn outl<'a, A, B>(pair: &'a (A, B)) -> &'a A {
    &pair.0
}

/// Projection onto the second component (outr).
pub fn outr<'a, A, B>(pair: &'a (A, B)) -> &'a B {
    &pair.1
}

/// Pairing combinator `(f, g)` built from functions that share their argument.
pub fn fork<'a, A, B, C, F, G>(f: F, g: G) -> impl Fn(&'a A) -> (B, C)
where
    F: Fn(&A) -> B,
    G: Fn(&A) -> C,
{
    move |value| (f(value), g(value))
}

/// Coproduct type capturing categorical sums.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sum<L, R> {
    Inl(L),
    Inr(R),
}

impl<L, R> Sum<L, R> {
    pub fn map<L2, R2, FL, FR>(self, mut left: FL, mut right: FR) -> Sum<L2, R2>
    where
        FL: FnMut(L) -> L2,
        FR: FnMut(R) -> R2,
    {
        match self {
            Sum::Inl(value) => Sum::Inl(left(value)),
            Sum::Inr(value) => Sum::Inr(right(value)),
        }
    }
}

/// (A + B) + C ≅ A + (B + C)
pub fn assoc_sum_left<A, B, C>(value: Sum<Sum<A, B>, C>) -> Sum<A, Sum<B, C>> {
    match value {
        Sum::Inl(Sum::Inl(a)) => Sum::Inl(a),
        Sum::Inl(Sum::Inr(b)) => Sum::Inr(Sum::Inl(b)),
        Sum::Inr(c) => Sum::Inr(Sum::Inr(c)),
    }
}

pub fn assoc_sum_right<A, B, C>(value: Sum<A, Sum<B, C>>) -> Sum<Sum<A, B>, C> {
    match value {
        Sum::Inl(a) => Sum::Inl(Sum::Inl(a)),
        Sum::Inr(Sum::Inl(b)) => Sum::Inl(Sum::Inr(b)),
        Sum::Inr(Sum::Inr(c)) => Sum::Inr(c),
    }
}

/// Distributes products over sums: (A + B) × C → (A × C) + (B × C)
pub fn distr_left<A, B, C>(value: (Sum<A, B>, C)) -> Sum<(A, C), (B, C)> {
    match value.0 {
        Sum::Inl(a) => Sum::Inl((a, value.1)),
        Sum::Inr(b) => Sum::Inr((b, value.1)),
    }
}

/// Inverse of `distr_left`.
pub fn distl_inv<A, B, C>(value: Sum<(A, C), (B, C)>) -> (Sum<A, B>, C) {
    match value {
        Sum::Inl((a, c)) => (Sum::Inl(a), c),
        Sum::Inr((b, c)) => (Sum::Inr(b), c),
    }
}

/// Case analysis `[f, g]` for coproducts.
pub fn cotuple<A, B, C, F, G>(mut f: F, mut g: G) -> impl FnMut(Sum<A, B>) -> C
where
    F: FnMut(A) -> C,
    G: FnMut(B) -> C,
{
    move |choice| match choice {
        Sum::Inl(value) => f(value),
        Sum::Inr(value) => g(value),
    }
}

/// Swaps the coordinates of a product (natural isomorphism between `A×B` and `B×A`).
pub fn swap<A, B>(pair: (A, B)) -> (B, A) {
    (pair.1, pair.0)
}

// -----------------------------------------------------------------------------
// Functors and catamorphisms (initial algebras)
// -----------------------------------------------------------------------------

/// A (covariant) endofunctor over the category of sets.
pub trait Functor {
    type Wrapped<T>;

    fn fmap<T, U, M>(value: Self::Wrapped<T>, mapper: M) -> Self::Wrapped<U>
    where
        M: FnMut(T) -> U;
}

/// Power-set mapping on concrete `HashSet`s.
pub fn power_map<T, U, M>(set: HashSet<T>, mut mapper: M) -> HashSet<U>
where
    T: Eq + Hash,
    U: Eq + Hash,
    M: FnMut(T) -> U,
{
    set.into_iter().map(|item| mapper(item)).collect()
}

/// Fixpoint of a functor – represents the carrier `T` of an initial `F`-algebra.
pub struct Fix<F>
where
    F: Functor,
{
    node: Box<F::Wrapped<Fix<F>>>,
}

impl<F> Fix<F>
where
    F: Functor,
{
    pub fn new(node: F::Wrapped<Fix<F>>) -> Self {
        Self {
            node: Box::new(node),
        }
    }

    pub fn into_inner(self) -> F::Wrapped<Fix<F>> {
        *self.node
    }
}

impl<F> fmt::Debug for Fix<F>
where
    F: Functor,
    F::Wrapped<Fix<F>>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt(f)
    }
}

/// Generic catamorphism – folds an initial algebra by replacing structure with `phi`.
pub fn cata<F, A, Phi>(term: Fix<F>, mut phi: Phi) -> A
where
    F: Functor,
    Phi: FnMut(F::Wrapped<A>) -> A,
{
    fn go<F, A, Phi>(term: Fix<F>, phi: &mut Phi) -> A
    where
        F: Functor,
        Phi: FnMut(F::Wrapped<A>) -> A,
    {
        let mapped = F::fmap(term.into_inner(), |child| go::<F, A, Phi>(child, phi));
        phi(mapped)
    }

    go::<F, A, Phi>(term, &mut phi)
}

// -----------------------------------------------------------------------------
// Natural numbers as an initial algebra of F(X) = 1 + X
// -----------------------------------------------------------------------------

pub mod nat {
    use super::{Fix, Functor, cata};

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum NatNode<R> {
        Zero,
        Succ(R),
    }

    pub struct NatF;

    impl Functor for NatF {
        type Wrapped<T> = NatNode<T>;

        fn fmap<T, U, M>(value: Self::Wrapped<T>, mut mapper: M) -> Self::Wrapped<U>
        where
            M: FnMut(T) -> U,
        {
            match value {
                NatNode::Zero => NatNode::Zero,
                NatNode::Succ(next) => NatNode::Succ(mapper(next)),
            }
        }
    }

    pub type Nat = Fix<NatF>;

    pub fn zero() -> Nat {
        Fix::new(NatNode::Zero)
    }

    pub fn succ(n: Nat) -> Nat {
        Fix::new(NatNode::Succ(n))
    }

    pub fn from_u64(n: u64) -> Nat {
        let mut acc = zero();
        for _ in 0..n {
            acc = succ(acc);
        }
        acc
    }

    pub fn fold<T, S>(value: Nat, zero_case: T, mut succ_case: S) -> T
    where
        S: FnMut(T) -> T,
    {
        let mut zero_slot = Some(zero_case);
        cata::<NatF, T, _>(value, |node| match node {
            NatNode::Zero => zero_slot
                .take()
                .expect("zero case evaluated more than once"),
            NatNode::Succ(acc) => succ_case(acc),
        })
    }

    pub fn to_u64(value: Nat) -> u64 {
        fold(value, 0u64, |acc| acc + 1)
    }

    pub fn add(a: Nat, b: Nat) -> Nat {
        fold(a, b, |acc| succ(acc))
    }
}

// -----------------------------------------------------------------------------
// Lists as an initial algebra of F(X) = 1 + (A × X)
// -----------------------------------------------------------------------------

pub mod list {
    use super::{Fix, Functor, cata};
    use std::marker::PhantomData;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum ListNode<A, R> {
        Nil,
        Cons(A, R),
    }

    pub struct ListF<A>(PhantomData<A>);

    impl<A> Functor for ListF<A> {
        type Wrapped<T> = ListNode<A, T>;

        fn fmap<T, U, M>(value: Self::Wrapped<T>, mut mapper: M) -> Self::Wrapped<U>
        where
            M: FnMut(T) -> U,
        {
            match value {
                ListNode::Nil => ListNode::Nil,
                ListNode::Cons(head, tail) => ListNode::Cons(head, mapper(tail)),
            }
        }
    }

    pub type List<A> = Fix<ListF<A>>;

    pub fn nil<A>() -> List<A> {
        Fix::new(ListNode::Nil)
    }

    pub fn cons<A>(head: A, tail: List<A>) -> List<A> {
        Fix::new(ListNode::Cons(head, tail))
    }

    pub fn from_vec<A>(values: Vec<A>) -> List<A> {
        values
            .into_iter()
            .rev()
            .fold(nil(), |acc, value| cons(value, acc))
    }

    pub fn fold<A, T, C>(list: List<A>, nil_value: T, mut cons_case: C) -> T
    where
        C: FnMut(A, T) -> T,
    {
        let mut base = Some(nil_value);
        cata::<ListF<A>, T, _>(list, |node| match node {
            ListNode::Nil => base.take().expect("nil case evaluated more than once"),
            ListNode::Cons(head, acc) => cons_case(head, acc),
        })
    }

    pub fn length<A>(list: List<A>) -> usize {
        fold(list, 0usize, |_, acc| acc + 1)
    }

    pub fn sum(list: List<u64>) -> u64 {
        fold(list, 0u64, |head, acc| head + acc)
    }

    pub fn map<A, B, F>(list: List<A>, mut f: F) -> List<B>
    where
        F: FnMut(A) -> B,
    {
        fold(list, nil(), |head, acc| cons(f(head), acc))
    }

    pub fn to_vec<A>(list: List<A>) -> Vec<A> {
        let mut result = fold(list, Vec::new(), |head, mut acc| {
            acc.insert(0, head);
            acc
        });
        result.shrink_to_fit();
        result
    }
}

// -----------------------------------------------------------------------------
// Tests aligning with the equational laws from Chapter 2
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_respects_projections() {
        let point = (3, 5);
        let diag = fork(|p: &(i32, i32)| p.0 + p.1, |p| p.0 * p.1);
        let result = diag(&point);
        assert_eq!(result.0, point.0 + point.1);
        assert_eq!(result.1, point.0 * point.1);
    }

    #[test]
    fn cotuple_routes_by_constructor() {
        let mut choose = cotuple(|x: i32| x + 1, |s: &str| s.len() as i32);
        assert_eq!(choose(Sum::Inl(4)), 5);
        assert_eq!(choose(Sum::Inr("abc")), 3);
    }

    #[test]
    fn coproduct_associativity_roundtrip() {
        let value: Sum<Sum<i32, i32>, i32> = Sum::Inl(Sum::Inr(10));
        let round = assoc_sum_right(assoc_sum_left(value));
        match round {
            Sum::Inl(Sum::Inr(v)) => assert_eq!(v, 10),
            _ => panic!("unexpected form"),
        }
    }

    #[test]
    fn distributivity_inverse() {
        let input: (Sum<&str, &str>, i32) = (Sum::Inr("x"), 5);
        let sum = distr_left(input);
        let round = distl_inv(sum);
        match round.0 {
            Sum::Inr(val) => {
                assert_eq!(val, "x");
                assert_eq!(round.1, 5);
            }
            _ => panic!("form"),
        }
    }

    #[test]
    fn power_set_functor_maps_elements() {
        let mut base: HashSet<i32> = HashSet::new();
        base.insert(1);
        base.insert(2);
        let mapped = power_map(base, |x| x * 2);
        assert!(mapped.contains(&2));
        assert!(mapped.contains(&4));
        assert_eq!(mapped.len(), 2);
    }

    #[test]
    fn natural_numbers_via_initial_algebra() {
        use nat::*;
        let five = from_u64(5);
        let eight = add(from_u64(5), from_u64(3));
        assert_eq!(to_u64(five), 5);
        assert_eq!(to_u64(eight), 8);
    }

    #[test]
    fn list_catamorphisms_match_expected_behaviour() {
        use list::*;
        let data = vec![1u64, 2, 3, 4];
        let list = from_vec(data.clone());
        assert_eq!(length(list), 4);
        let list = from_vec(data.clone());
        assert_eq!(sum(list), 10);
        let doubled = map(from_vec(data.clone()), |x| x * 2);
        assert_eq!(to_vec(doubled), vec![2, 4, 6, 8]);
        assert_eq!(to_vec(from_vec(data.clone())), data);
    }
}
