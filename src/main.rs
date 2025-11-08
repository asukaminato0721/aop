mod chapter1;
mod chapter2;

use chapter1::{BoolValue, Both, Nat, ackermann, factorial, fibonacci, plus, switch_bool};
use chapter2::{list, nat, swap};

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
}
