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

use std::cmp::max;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use rand_distr::num_traits::real::Real;
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

    fn max_number_pulls(&self) -> i32 {
        let mut max_number_pulls = 0;
        for arm in &self.arm_memory {
            if arm.get_num_pulls() > max_number_pulls {
                max_number_pulls = arm.get_num_pulls();
            }
        }
        max_number_pulls
    }

    fn find_best_ucb(&self) -> i32 {
        let arm_index_ucb_norm_min: i32 = *self.sample_average_tree.iter().next().unwrap().1;
        let ucb_norm_min: f64 = self.arm_memory[arm_index_ucb_norm_min as usize].get_mean_reward();

        let max_number_pulls = self.max_number_pulls();

        let mut ucb_norm_max: f64 = ucb_norm_min;

        for (ucb_norm, arm_index) in self.sample_average_tree.iter() {
            ucb_norm_max = f64::max(ucb_norm_max, self.arm_memory[*arm_index as usize].get_mean_reward());

            // checks if we are still in the non dominated-set (current mean <= mean_max_pulls)
            if self.arm_memory[*arm_index as usize].get_num_pulls() == max_number_pulls {
                break;
            }
        }

        // find the solution of non-dominated set with the lowest associated UCB value
        let mut best_arm_index: i32 = 0;
        let mut best_ucb_value: f64 = f64::MAX;


        for (ucb_norm, arm_index) in self.sample_average_tree.iter() {
            if ucb_norm_max == ucb_norm_min {
                best_arm_index = *arm_index;
            }

            // transform sample mean to interval [0,1]
            let transformed_sample_mean: f64 = (self.arm_memory[*arm_index as usize].get_mean_reward() - ucb_norm_min) / (ucb_norm_max - ucb_norm_min);
            let penalty_term: f64 = (2.0 * (self.genetic_algorithm.get_simulations_used() as f64).ln() as f64 / self.arm_memory[*arm_index as usize].get_num_pulls() as f64).sqrt();
            let ucb_value: f64 = transformed_sample_mean + penalty_term;

            // new best solution found
            if ucb_value < best_ucb_value {
                best_arm_index = *arm_index;
                best_ucb_value = ucb_value;
            }

            // checks if we are still in the non dominated-set (current mean <= mean_max_pulls)
            if self.arm_memory[*arm_index as usize].get_num_pulls() == max_number_pulls {
                break;
            }
        }


        best_arm_index
    }

    fn sample_and_update(&mut self, arm_index: i32, mut individual: Arm) {
        if arm_index >= 0 {
            self.delete_sample_average_tree_node(&individual, arm_index);
            self.arm_memory[arm_index as usize].pull();
            self.genetic_algorithm.update_simulations_used(1);
            self.sample_average_tree.insert(FloatKey(self.arm_memory[arm_index as usize].get_mean_reward()), arm_index);
        } else {
            individual.pull();
            self.genetic_algorithm.update_simulations_used(1);
            self.arm_memory.push(individual.clone());
            self.lookup_tabel.insert(individual.get_action_vector(), self.arm_memory.len() as i32 - 1);
            self.sample_average_tree.insert(FloatKey(individual.get_mean_reward()), self.arm_memory.len() as i32 - 1);
        }
    }

    fn optimize(&mut self) -> Vec<i32> {

        loop {
            let crossover_pop = self.genetic_algorithm.crossover();

            // mutate automatically removes duplicates
            let mutated_pop = self.genetic_algorithm.mutate(crossover_pop);


            for individual_index in 0..mutated_pop.len() {

                let arm_index: i32 = self.lookup_tabel.get(&mutated_pop[individual_index].get_action_vector()).unwrap().clone();
                self.sample_and_update(arm_index, mutated_pop[individual_index].clone());
                if self.genetic_algorithm.budget_reached() {
                    return self.arm_memory[self.find_best_ucb() as usize].get_action_vector();
                }
            }

            let individuals = self.genetic_algorithm.get_individuals().clone();

            for individual in individuals {
                let arm_index: i32 = self.lookup_tabel.get(&individual.get_action_vector()).unwrap().clone();
                self.sample_and_update(arm_index, individual.clone());
                if self.genetic_algorithm.budget_reached() {
                    return self.arm_memory[self.find_best_ucb() as usize].get_action_vector();
                }
            }


        }

    }
}
