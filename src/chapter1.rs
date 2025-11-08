//! Chapter 1: Programs — Rust translations of the Algebra of Programming examples.
//!
//! The original text develops datatypes (booleans, numbers, lists, trees), the
//! associated fold operators, and a catalogue of staple functions.  This module
//! mirrors those definitions in idiomatic Rust.  Each section groups the data
//! declarations and equations that appeared in the Haskell-style pseudocode and
//! rewrites them as concrete, testable Rust implementations.
#![allow(dead_code)]

use std::fmt;
use std::ops::{Add, Mul};

// -----------------------------------------------------------------------------
// 1.1 Datatypes
// -----------------------------------------------------------------------------

/// Boolean values used throughout the chapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoolValue {
    False,
    True,
}

impl BoolValue {
    pub const fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::True => Self::False,
        }
    }

    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, Self::True) => Self::True,
            _ => Self::False,
        }
    }
}

/// Convenience tuple wrapper used for functions such as `switch`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Both<A, B>(pub A, pub B);

impl<A, B> Both<A, B> {
    pub fn map_first<F, R>(self, f: F) -> Both<R, B>
    where
        F: FnOnce(A) -> R,
    {
        Both(f(self.0), self.1)
    }
}

/// Swap the boolean component using logical negation.
pub fn switch_bool<T>(value: Both<BoolValue, T>) -> Both<BoolValue, T> {
    value.map_first(|b| b.not())
}

/// Non-curried conjunction operating on a pair.
pub fn and_pair(args: Both<BoolValue, BoolValue>) -> BoolValue {
    args.0.and(args.1)
}

/// Curried conjunction, matching the `cand` definition.
pub fn cand(lhs: BoolValue) -> impl Fn(BoolValue) -> BoolValue {
    move |rhs| lhs.and(rhs)
}

/// Simple Maybe type (called `maybe A ::= nothing | just A` in the book).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Maybe<T> {
    Nothing,
    Just(T),
}

impl<T> Maybe<T> {
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Maybe<U> {
        match self {
            Maybe::Nothing => Maybe::Nothing,
            Maybe::Just(value) => Maybe::Just(f(value)),
        }
    }
}

impl<T> From<Option<T>> for Maybe<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => Maybe::Just(v),
            None => Maybe::Nothing,
        }
    }
}

impl<T> From<Maybe<T>> for Option<T> {
    fn from(value: Maybe<T>) -> Self {
        match value {
            Maybe::Nothing => None,
            Maybe::Just(v) => Some(v),
        }
    }
}

/// Curry a function of pairs into a function returning functions.
pub fn curry<A, B, C, F>(f: F) -> impl Fn(A) -> Box<dyn Fn(B) -> C>
where
    A: Clone + 'static,
    B: 'static,
    C: 'static,
    F: Fn((A, B)) -> C + Clone + 'static,
{
    move |a: A| {
        let f_clone = f.clone();
        let captured = a.clone();
        Box::new(move |b: B| f_clone((captured.clone(), b)))
    }
}

/// The inverse of `curry`.
pub fn uncurry<A, B, C, F, G>(f: F) -> impl Fn((A, B)) -> C
where
    F: Fn(A) -> G,
    G: Fn(B) -> C,
{
    move |(a, b)| {
        let inner = f(a);
        inner(b)
    }
}

// -----------------------------------------------------------------------------
// 1.2 Natural numbers
// -----------------------------------------------------------------------------

/// Peano-style natural numbers backed by `u64` but accompanied by fold-based
/// operations to mirror the structural recursion from the text.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Nat(pub u64);

impl Nat {
    pub const ZERO: Nat = Nat(0);

    pub fn succ(self) -> Self {
        Nat(self.0 + 1)
    }

    pub const fn value(self) -> u64 {
        self.0
    }

    /// `foldn (c, h)` in the book.
    pub fn fold<T, F>(self, zero_case: T, succ_case: F) -> T
    where
        T: Clone,
        F: Fn(T) -> T,
    {
        let mut acc = zero_case;
        for _ in 0..self.0 {
            acc = succ_case(acc);
        }
        acc
    }
}

impl From<u64> for Nat {
    fn from(value: u64) -> Self {
        Nat(value)
    }
}

impl fmt::Debug for Nat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Nat {
    type Output = Nat;
    fn add(self, rhs: Nat) -> Self::Output {
        Nat(self.0 + rhs.0)
    }
}

impl Mul for Nat {
    type Output = Nat;
    fn mul(self, rhs: Nat) -> Self::Output {
        Nat(self.0 * rhs.0)
    }
}

pub fn plus(m: Nat, n: Nat) -> Nat {
    n.fold(m, |acc| acc.succ())
}

pub fn mult(m: Nat, n: Nat) -> Nat {
    n.fold(Nat::ZERO, |acc| plus(acc, m))
}

pub fn exp(base: Nat, exponent: Nat) -> Nat {
    exponent.fold(Nat::from(1), |acc| mult(acc, base))
}

pub fn factorial(n: Nat) -> Nat {
    let pair = n.fold((Nat::ZERO, Nat::from(1)), |(m, acc)| {
        let next = m.succ();
        (next, mult(next, acc))
    });
    pair.1
}

pub fn fibonacci(n: Nat) -> Nat {
    let pair = n.fold((Nat::ZERO, Nat::from(1)), |(a, b)| (b, plus(a, b)));
    pair.0
}

pub fn ackermann(m: Nat, n: Nat) -> Nat {
    fn inner(m: u64, n: u64) -> u64 {
        if m == 0 {
            n + 1
        } else if n == 0 {
            inner(m - 1, 1)
        } else {
            let temp = inner(m, n - 1);
            inner(m - 1, temp)
        }
    }

    Nat(inner(m.value(), n.value()))
}

// -----------------------------------------------------------------------------
// 1.3 Lists (cons- and snoc-style), folds, and derived functions
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConsList<T> {
    Nil,
    Cons(T, Box<ConsList<T>>),
}

impl<T> ConsList<T> {
    pub fn nil() -> Self {
        Self::Nil
    }

    pub fn cons(head: T, tail: Self) -> Self {
        Self::Cons(head, Box::new(tail))
    }

    pub fn foldr<U, F>(&self, nil_case: U, cons_case: &F) -> U
    where
        F: Fn(&T, U) -> U,
    {
        match self {
            ConsList::Nil => nil_case,
            ConsList::Cons(head, tail) => {
                let acc = tail.foldr(nil_case, cons_case);
                cons_case(head, acc)
            }
        }
    }

    pub fn map<U, F>(&self, f: &F) -> ConsList<U>
    where
        F: Fn(&T) -> U,
    {
        self.foldr(ConsList::Nil, &|value, acc| ConsList::cons(f(value), acc))
    }

    pub fn concat(&self, other: &ConsList<T>) -> ConsList<T>
    where
        T: Clone,
    {
        self.foldr(other.clone(), &|value, acc| {
            ConsList::cons(value.clone(), acc)
        })
    }

    pub fn length(&self) -> Nat {
        self.foldr(Nat::ZERO, &|_, acc| acc.succ())
    }

    pub fn reverse(&self) -> ConsList<T>
    where
        T: Clone,
    {
        self.foldr(ConsList::Nil, &|value, acc| snocr_cons(acc, value.clone()))
    }

    pub fn filter<F>(&self, predicate: &F) -> ConsList<T>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        self.foldr(ConsList::Nil, &|value, acc| {
            if predicate(value) {
                ConsList::cons(value.clone(), acc)
            } else {
                acc
            }
        })
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        let mut current = self.clone();
        while let ConsList::Cons(head, tail) = current {
            result.push(head);
            current = *tail;
        }
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnocList<T> {
    Nil,
    Snoc(Box<SnocList<T>>, T),
}

impl<T> SnocList<T> {
    pub fn nil() -> Self {
        Self::Nil
    }

    pub fn snoc(self, value: T) -> Self {
        SnocList::Snoc(Box::new(self), value)
    }

    pub fn foldl<U, F>(&self, nil_case: U, snoc_case: &F) -> U
    where
        F: Fn(U, &T) -> U,
    {
        match self {
            SnocList::Nil => nil_case,
            SnocList::Snoc(prefix, value) => {
                let acc = prefix.foldl(nil_case, snoc_case);
                snoc_case(acc, value)
            }
        }
    }

    pub fn map<U, F>(&self, f: &F) -> SnocList<U>
    where
        U: Clone,
        F: Fn(&T) -> U,
    {
        self.foldl(SnocList::Nil, &|acc, value| acc.snoc(f(value)))
    }

    pub fn concat(&self, other: &SnocList<T>) -> SnocList<T>
    where
        T: Clone,
    {
        other.foldl(self.clone(), &|acc, value| acc.snoc(value.clone()))
    }
}

pub fn snocr_cons<T>(list: ConsList<T>, value: T) -> ConsList<T> {
    match list {
        ConsList::Nil => ConsList::cons(value, ConsList::Nil),
        ConsList::Cons(head, tail) => ConsList::cons(head, snocr_cons(*tail, value)),
    }
}

pub fn convert_snoc_to_cons<T: Clone>(list: &SnocList<T>) -> ConsList<T> {
    match list {
        SnocList::Nil => ConsList::Nil,
        SnocList::Snoc(prefix, value) => {
            let converted = convert_snoc_to_cons(prefix);
            snocr_cons(converted, value.clone())
        }
    }
}

pub fn cat_snoc<T: Clone>(x: &SnocList<T>, y: &SnocList<T>) -> SnocList<T> {
    y.foldl(x.clone(), &|acc, value| acc.snoc(value.clone()))
}

pub fn sum(list: &ConsList<Nat>) -> Nat {
    list.foldr(Nat::ZERO, &|value, acc| plus(*value, acc))
}

pub fn product(list: &ConsList<Nat>) -> Nat {
    list.foldr(Nat::from(1), &|value, acc| mult(*value, acc))
}

pub fn concat_lists<T: Clone>(lists: &ConsList<ConsList<T>>) -> ConsList<T> {
    lists.foldr(ConsList::Nil, &|value, acc| value.clone().concat(&acc))
}

pub fn length<T>(list: &ConsList<T>) -> Nat {
    list.length()
}

pub fn filter_list<T: Clone, F>(list: &ConsList<T>, predicate: &F) -> ConsList<T>
where
    F: Fn(&T) -> bool,
{
    list.filter(predicate)
}

pub fn reverse_cons_list<T: Clone>(list: &ConsList<T>) -> ConsList<T> {
    list.reverse()
}

pub fn reverse_snoc_list<T: Clone>(list: &SnocList<T>) -> SnocList<T> {
    list.foldl(SnocList::Nil, &|acc, value| {
        SnocList::Snoc(Box::new(acc), value.clone())
    })
}

pub fn eval_decimal(int_part: &SnocList<u32>, frac_part: &ConsList<u32>) -> f64 {
    let whole = int_part.foldl(0.0, &|acc, digit| acc * 10.0 + f64::from(*digit));
    let frac = frac_part.foldr(0.0, &|digit, acc| (f64::from(*digit) + acc) / 10.0);
    whole + frac
}

// Non-empty lists -------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NonEmptyConsList<T> {
    Wrap(T),
    Cons(T, Box<NonEmptyConsList<T>>),
}

impl<T> NonEmptyConsList<T> {
    pub fn fold<U, F, G>(&self, wrap_case: &F, cons_case: &G) -> U
    where
        F: Fn(&T) -> U,
        G: Fn(&T, U) -> U,
    {
        match self {
            NonEmptyConsList::Wrap(value) => wrap_case(value),
            NonEmptyConsList::Cons(value, tail) => {
                let acc = tail.fold(wrap_case, cons_case);
                cons_case(value, acc)
            }
        }
    }

    pub fn head(&self) -> &T {
        match self {
            NonEmptyConsList::Wrap(value) => value,
            NonEmptyConsList::Cons(value, _) => value,
        }
    }
}

// -----------------------------------------------------------------------------
// 1.4 Trees and folds
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryTree<T> {
    Tip(T),
    Bin(Box<BinaryTree<T>>, Box<BinaryTree<T>>),
}

impl<T> BinaryTree<T> {
    pub fn fold<U, F, G>(&self, tip_case: &F, bin_case: &G) -> U
    where
        F: Fn(&T) -> U,
        G: Fn(U, U) -> U,
    {
        match self {
            BinaryTree::Tip(value) => tip_case(value),
            BinaryTree::Bin(left, right) => {
                let l = left.fold(tip_case, bin_case);
                let r = right.fold(tip_case, bin_case);
                bin_case(l, r)
            }
        }
    }

    pub fn size(&self) -> Nat {
        self.fold(&|_| Nat::from(1), &|l, r| plus(l, r))
    }

    pub fn depth(&self) -> Nat {
        self.fold(&|_| Nat::from(1), &|l, r| {
            let max = if l.value() > r.value() { l } else { r };
            max.succ()
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoseTree<T> {
    Node(T, Vec<RoseTree<T>>),
}

impl<T> RoseTree<T> {
    pub fn fold<U, F, G>(&self, node_case: &F, combine: &G) -> U
    where
        F: Fn(&T, Vec<U>) -> U,
        G: Fn(U, U) -> U,
        U: Clone,
    {
        match self {
            RoseTree::Node(value, children) => {
                let mut mapped = Vec::with_capacity(children.len());
                for child in children {
                    mapped.push(child.fold(node_case, combine));
                }
                node_case(value, mapped)
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 1.5 Inverses (zip / unzip) and related helpers
// -----------------------------------------------------------------------------

pub fn pair<F, G, A, B, C>(f: F, g: G) -> impl Fn(A) -> (B, C)
where
    F: Fn(&A) -> B,
    G: Fn(&A) -> C,
    A: Clone,
{
    move |value: A| (f(&value), g(&value))
}

pub fn cross<F, G, A, B, C, D>(f: F, g: G) -> impl Fn((A, B)) -> (C, D)
where
    F: Fn(A) -> C,
    G: Fn(B) -> D,
{
    move |(a, b)| (f(a), g(b))
}

pub fn zip_lists<A: Clone, B: Clone>(xs: &ConsList<A>, ys: &ConsList<B>) -> ConsList<(A, B)> {
    match (xs, ys) {
        (ConsList::Cons(ax, rest_x), ConsList::Cons(by, rest_y)) => {
            ConsList::cons((ax.clone(), by.clone()), zip_lists(rest_x, rest_y))
        }
        _ => ConsList::Nil,
    }
}

pub fn unzip_pairs<A: Clone, B: Clone>(pairs: &ConsList<(A, B)>) -> (ConsList<A>, ConsList<B>) {
    pairs.foldr((ConsList::Nil, ConsList::Nil), &|(a, b), (xs, ys)| {
        (ConsList::cons(a.clone(), xs), ConsList::cons(b.clone(), ys))
    })
}

// -----------------------------------------------------------------------------
// 1.6 Polymorphic/natural behaviour examples
// -----------------------------------------------------------------------------

pub fn inits<T: Clone>(list: &SnocList<T>) -> SnocList<SnocList<T>> {
    list.foldl(
        SnocList::Snoc(Box::new(SnocList::Nil), SnocList::Nil),
        &|acc, value| {
            let (prefixes, last) = match acc {
                SnocList::Snoc(prefixes, last) => (*prefixes, last),
                SnocList::Nil => unreachable!("constructed with Snoc"),
            };
            let extended_last = last.clone().snoc(value.clone());
            let updated_prefixes = prefixes.snoc(extended_last.clone());
            SnocList::Snoc(Box::new(updated_prefixes), extended_last)
        },
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_ops() {
        assert_eq!(BoolValue::True.not(), BoolValue::False);
        assert_eq!(
            and_pair(Both(BoolValue::True, BoolValue::False)),
            BoolValue::False
        );
    }

    #[test]
    fn curry_uncurry_roundtrip() {
        let f = |(a, b): (i32, i32)| a + b;
        let curried = curry(f.clone());
        let add_two = curried(2);
        assert_eq!(add_two(3), 5);
        let uncurried = uncurry(curried);
        assert_eq!(uncurried((4, 7)), 11);
    }

    #[test]
    fn nat_arithmetic() {
        let two = Nat::from(2);
        let three = Nat::from(3);
        assert_eq!(plus(two, three).value(), 5);
        assert_eq!(mult(two, three).value(), 6);
        assert_eq!(exp(two, three).value(), 8);
        assert_eq!(factorial(three).value(), 6);
        assert_eq!(fibonacci(Nat::from(7)).value(), 13);
    }

    #[test]
    fn ackermann_small_inputs() {
        assert_eq!(ackermann(Nat::from(2), Nat::from(2)).value(), 7);
    }

    fn list_of_ints(values: &[i32]) -> ConsList<i32> {
        let mut result = ConsList::Nil;
        for value in values.iter().rev() {
            result = ConsList::cons(*value, result);
        }
        result
    }

    #[test]
    fn cons_list_operations() {
        let xs = list_of_ints(&[1, 2, 3]);
        let ys = list_of_ints(&[4, 5]);
        let concatenated = xs.concat(&ys);
        assert_eq!(concatenated.to_vec(), vec![1, 2, 3, 4, 5]);
        let reversed = concatenated.reverse();
        assert_eq!(reversed.to_vec(), vec![5, 4, 3, 2, 1]);
        let filtered = reverse_cons_list(&reversed).filter(&|v| *v % 2 == 1);
        assert_eq!(filtered.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn zip_unzip_roundtrip() {
        let xs = list_of_ints(&[1, 2, 3]);
        let ys = list_of_ints(&[10, 20, 30]);
        let zipped = zip_lists(&xs, &ys);
        let (unzipped_xs, unzipped_ys) = unzip_pairs(&zipped);
        assert_eq!(unzipped_xs.to_vec(), vec![1, 2, 3]);
        assert_eq!(unzipped_ys.to_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn binary_tree_fold_examples() {
        let tree = BinaryTree::Bin(
            Box::new(BinaryTree::Tip(1)),
            Box::new(BinaryTree::Bin(
                Box::new(BinaryTree::Tip(2)),
                Box::new(BinaryTree::Tip(3)),
            )),
        );
        assert_eq!(tree.size().value(), 3);
        assert_eq!(tree.depth().value(), 3);
    }
}
