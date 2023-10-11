mod arm;
mod genetic;

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

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use multimap::MultiMap;
use arm::Arm;
use genetic::GeneticAlgorithm;

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct FloatKey(f64);

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for FloatKey {
    fn from(item: f64) -> Self {
        FloatKey(item)
    }
}

impl From<FloatKey> for f64 {
    fn from(item: FloatKey) -> f64 {
        item.0
    }
}

struct Gmab {
    sample_average_tree: MultiMap<FloatKey, i32>,
    arm_memory: Vec<Arm>,
    lookup_tabel: HashMap<Vec<i32>, i32>,
    genetic_algorithm: GeneticAlgorithm,

}

impl Gmab {
    fn get_arm_index(&self, individual: &Arm) -> i32 {
        match self.lookup_tabel.get(&individual.get_action_vector()) {
            Some(&index) => index,
            None => -1,
        }
    }

    fn delete_sample_average_tree_node(&mut self, individual: &arm::Arm, arm_index: i32) {
        let mean_reward = FloatKey(individual.get_mean_reward());

        if self.sample_average_tree.contains_key(&mean_reward) {
            let entries = self.sample_average_tree.get_vec_mut(&mean_reward).unwrap();
            if let Some(pos) = entries.iter().position(|&x| x == arm_index) {
                entries.remove(pos);
            }
            if entries.is_empty() {
                self.sample_average_tree.remove(&mean_reward);
            }
        }
    }

    fn new(
        opti_function: fn(Vec<i32>) -> f64,
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        mutation_span: f64,
        max_simulations: i32,
        dimension: usize,
        lower_bound: Vec<i32>,
        upper_bound: Vec<i32>, ) -> Gmab {

        let mut genetic_algorithm = GeneticAlgorithm::new(
            opti_function,
            population_size,
            mutation_rate,
            crossover_rate,
            mutation_span,
            max_simulations,
            dimension,
            lower_bound,
            upper_bound,
        );

        let mut arm_memory: Vec<Arm> = Vec::new();
        let mut lookup_tabel: HashMap<Vec<i32>, i32> = HashMap::new();
        let mut sample_average_tree: MultiMap<FloatKey, i32> = MultiMap::new();

        for (index, individual) in genetic_algorithm.get_individuals().iter_mut().enumerate() {
            individual.pull();
            arm_memory.push(individual.clone());
            lookup_tabel.insert(individual.get_action_vector(), index as i32);
            sample_average_tree.insert(FloatKey(individual.get_mean_reward()), index as i32);
        }

        Gmab {
            sample_average_tree,
            arm_memory,
            lookup_tabel,
            genetic_algorithm,
        }
    }

    fn find_best_ucb(&self) -> i32 {
        let arm_index_ucb_norm_min: i32 = *self.sample_average_tree.iter().next().unwrap().1;
        arm_index_ucb_norm_min
    }
}
