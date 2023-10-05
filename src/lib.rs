mod arm;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn some_function(inventory_fn: fn(Vec<i32>) -> f64, action_vector: Vec<i32>) {
    let result = inventory_fn(action_vector);
    println!("Result from inventory function: {}", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
