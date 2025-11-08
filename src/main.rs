mod chapter1;

use chapter1::{BoolValue, Both, Nat, ackermann, factorial, fibonacci, plus, switch_bool};

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
}
