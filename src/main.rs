mod chapter1;
mod chapter2;
mod chapter3;
mod chapter4;

use chapter1::{BoolValue, Both, Nat, ackermann, factorial, fibonacci, plus, switch_bool};
use chapter2::{list, nat, swap};
use chapter3::{average, banana_split_stats, intern_decimal, triangle_fn};
use chapter4::{Function as RelFunction, Relation};
use std::collections::HashSet;

fn main() {
    let result = plus(Nat::from(2), Nat::from(3));
    println!("2 + 3 = {}", result.value());

    let fib10 = fibonacci(Nat::from(10));
    println!("fib 10 = {}", fib10.value());

    let ack = ackermann(Nat::from(2), Nat::from(2));
    println!("ack 2 2 = {}", ack.value());

    let flipped = switch_bool(Both(BoolValue::True, "chapter 1"));
    println!("switch Bool True -> {:?}", flipped.0);

    println!("10! = {}", factorial(Nat::from(10)).value());

    let nat_eight = nat::from_u64(8);
    println!("chapter 2 nat 8 -> {}", nat::to_u64(nat_eight));

    let list_values = vec![1u64, 2, 3, 4];
    println!(
        "list length {}, sum {}",
        list::length(list::from_vec(list_values.clone())),
        list::sum(list::from_vec(list_values.clone()))
    );

    let swapped = swap(("left", 42));
    println!("swap (left, 42) -> ({}, {})", swapped.0, swapped.1);

    let stats = banana_split_stats(list::from_vec(list_values.clone()));
    println!("banana-split stats sum {} len {}", stats.0, stats.1);
    if let Some(avg) = average(list::from_vec(list_values.clone())) {
        println!("average {:?}", avg);
    }

    let triangle = triangle_fn(list::from_vec(vec![1, 2, 3]), |x| x + 1);
    println!("triangle succ {:?}", list::to_vec(triangle));

    let tex_digits = vec![1, 2, 5];
    println!(
        "intern decimal {:?} -> {}",
        tex_digits,
        intern_decimal(&tex_digits)
    );

    let relation = Relation::from_pairs(vec![
        ("approved".to_string(), "design".to_string()),
        ("approved".to_string(), "spec".to_string()),
        ("draft".to_string(), "design".to_string()),
    ]);
    let statuses: HashSet<_> = relation.left_values();
    println!("range size {}", statuses.len());
    let tab = relation.tabulate();
    let routing = RelFunction::from_pairs(vec![("alpha", 0usize), ("beta", 2usize)]);
    let routed_status = tab.left().after(&routing);
    if let Some(val) = routed_status.apply(&"alpha") {
        println!("tabulated alpha -> {}", val);
    }
}
