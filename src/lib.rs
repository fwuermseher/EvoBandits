mod arm;
mod genetic;


use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use arm::Arm;
use genetic::GeneticAlgorithm;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
struct FloatKey(f64);

impl Eq for FloatKey {}

impl Ord for FloatKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

#[derive(Debug)]
struct SortedMultiMap<K: Ord, V: PartialEq> {
    inner: BTreeMap<K, Vec<V>>,
}

impl<K: Ord, V: PartialEq> SortedMultiMap<K, V> {
    pub fn new() -> Self {
        SortedMultiMap { inner: BTreeMap::new() }
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.inner.entry(key).or_insert_with(Vec::new).push(value);
    }

    pub fn delete(&mut self, key: &K, value: &V) -> bool {
        if let Some(values) = self.inner.get_mut(key) {
            if let Some(pos) = values.iter().position(|v| v == value) {
                values.remove(pos);
                if values.is_empty() {
                    self.inner.remove(key);
                }
                return true;
            }
        }
        false
    }

    pub fn iter(&self) -> impl Iterator<Item=(&K, &V)> {
        self.inner.iter().flat_map(|(key, values)| {
            values.iter().map(move |value| (key, value))
        })
    }
}

pub struct Gmab {
    sample_average_tree: SortedMultiMap<FloatKey, i32>,
    arm_memory: Vec<Arm>,
    lookup_tabel: HashMap<Vec<i32>, i32>,
    genetic_algorithm: GeneticAlgorithm,
    current_indexes: Vec<i32>,
}

impl Gmab {
    fn get_arm_index(&self, individual: &Arm) -> i32 {
        match self.lookup_tabel.get(&individual.get_action_vector().to_vec()) {
            Some(&index) => index,
            None => -1,
        }
    }

    pub fn new(
        opti_function: fn(&[i32]) -> f64,
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
        let mut sample_average_tree: SortedMultiMap<FloatKey, i32> = SortedMultiMap::new();

        for (index, individual) in genetic_algorithm.get_individuals().iter_mut().enumerate() {
            individual.pull();
            arm_memory.push(individual.clone());
            lookup_tabel.insert(individual.get_action_vector().to_vec(), index as i32);
            sample_average_tree.insert(FloatKey(individual.get_mean_reward()), index as i32);
        }

        Gmab {
            sample_average_tree,
            arm_memory,
            lookup_tabel,
            genetic_algorithm,
            current_indexes: Vec::new(),
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

        for (_ucb_norm, arm_index) in self.sample_average_tree.iter() {
            ucb_norm_max = f64::max(ucb_norm_max, self.arm_memory[*arm_index as usize].get_mean_reward());

            // checks if we are still in the non dominated-set (current mean <= mean_max_pulls)
            if self.arm_memory[*arm_index as usize].get_num_pulls() == max_number_pulls {
                break;
            }
        }

        // find the solution of non-dominated set with the lowest associated UCB value
        let mut best_arm_index: i32 = 0;
        let mut best_ucb_value: f64 = f64::MAX;


        for (_ucb_norm, arm_index) in self.sample_average_tree.iter() {
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
            self.sample_average_tree.delete(&FloatKey(self.arm_memory[arm_index as usize].get_mean_reward()), &arm_index);
            self.arm_memory[arm_index as usize].pull();
            self.genetic_algorithm.update_simulations_used(1);
            self.sample_average_tree.insert(FloatKey(self.arm_memory[arm_index as usize].get_mean_reward()), arm_index);
        } else {
            individual.pull();
            self.genetic_algorithm.update_simulations_used(1);
            self.arm_memory.push(individual.clone());
            self.lookup_tabel.insert(individual.get_action_vector().to_vec(), self.arm_memory.len() as i32 - 1);
            self.sample_average_tree.insert(FloatKey(individual.get_mean_reward()), self.arm_memory.len() as i32 - 1);
        }
    }

    pub fn optimize(&mut self, verbose: bool) -> Vec<i32> {
        loop {
            self.genetic_algorithm.get_individuals().clear();
            self.current_indexes.clear();

            // get first self.population_size elements from sorted tree and use value to get arm
            self.sample_average_tree.iter().take(self.genetic_algorithm.get_population_size()).for_each(|(_key, arm_index)| {
                self.genetic_algorithm.get_individuals().push(self.arm_memory[*arm_index as usize].clone());
                self.current_indexes.push(*arm_index);
            });

            // shuffle population
            self.genetic_algorithm.shuffle_population();


            let crossover_pop = self.genetic_algorithm.crossover();

            // mutate automatically removes duplicates
            let mutated_pop = self.genetic_algorithm.mutate(&crossover_pop);


            for individual_index in 0..mutated_pop.len() {

                let arm_index = self.get_arm_index(&mutated_pop[individual_index]);

                // check if arm is in current population
                if self.current_indexes.contains(&arm_index) {
                    continue;
                }

                self.sample_and_update(arm_index, mutated_pop[individual_index].clone());

                if self.genetic_algorithm.budget_reached() {
                    return self.arm_memory[self.find_best_ucb() as usize].get_action_vector().to_vec();
                }
            }

            let individuals = self.genetic_algorithm.get_individuals().clone();

            for individual in individuals {

                let arm_index = self.get_arm_index(&individual);
                self.sample_and_update(arm_index, individual.clone());

                if self.genetic_algorithm.budget_reached() {
                    return self.arm_memory[self.find_best_ucb() as usize].get_action_vector().to_vec();
                }
            }

            if verbose {
                let best_arm_index = self.find_best_ucb();
                print!("x: {:?}", self.arm_memory[best_arm_index as usize].get_action_vector());
                // get averaged function value over 50 simulations
                let mut sum = 0.0;
                for _ in 0..50 {
                    sum += self.arm_memory[best_arm_index as usize].get_function_value();
                }
                print!(" f(x): {:.3}", sum / 50.0);

                print!(" n: {}", self.genetic_algorithm.get_simulations_used());
                // print number of pulls of best arm
                println!(" n(x): {}", self.arm_memory[best_arm_index as usize].get_num_pulls());
            }
        }
    }
}
