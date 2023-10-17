use std::collections::HashSet;
use crate::arm::Arm;
use rand_distr::{Normal, Distribution};
use rand::Rng;


pub(crate) struct GeneticAlgorithm {
    mutation_rate: f64,
    crossover_rate: f64,
    mutation_span: f64,
    population_size: usize,
    individuals: Vec<Arm>,
    pub(crate) opti_function: fn(Vec<i32>) -> f64,
    max_simulations: i32,
    dimension: usize,
    lower_bound: Vec<i32>,
    upper_bound: Vec<i32>,
    simulations_used: i32,
}

impl GeneticAlgorithm {
    pub(crate) fn get_population_size(&self) -> usize {
        return self.population_size;
    }

    pub(crate) fn get_individuals(&mut self) ->  &mut Vec<Arm> {
        return &mut self.individuals;
    }

    pub(crate) fn get_simulations_used(&self) -> i32 {
        return self.simulations_used;
    }

    pub(crate) fn update_simulations_used(&mut self, number_of_new_simulations: i32) {
        self.simulations_used += number_of_new_simulations;
    }

    pub(crate) fn budget_reached(&self) -> bool {
        return self.simulations_used >= self.max_simulations;
    }

    pub(crate) fn new(
        opti_function: fn(Vec<i32>) -> f64,
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        mutation_span: f64,
        max_simulations: i32,
        dimension: usize,
        lower_bound: Vec<i32>,
        upper_bound: Vec<i32>,
    ) -> GeneticAlgorithm {
        let mut individuals: Vec<Arm> = Vec::new();
        let mut init_solutions: Vec<Vec<i32>> = Vec::new();

        for _ in 0..population_size {
            let action_vector = GeneticAlgorithm::generate_unique_solution(&init_solutions, &lower_bound, &upper_bound, dimension);
            init_solutions.push(action_vector);
            individuals.push(Arm::new(opti_function, init_solutions.last().unwrap().clone()));
        }

        GeneticAlgorithm {
            mutation_rate,
            crossover_rate,
            mutation_span,
            population_size,
            individuals,
            opti_function,
            max_simulations,
            dimension,
            lower_bound,
            upper_bound,
            simulations_used: 0,
        }
    }

    fn generate_unique_solution(
        solutions: &Vec<Vec<i32>>,
        lower_bound: &Vec<i32>,
        upper_bound: &Vec<i32>,
        dimension: usize,
    ) -> Vec<i32> {
        let mut rng = rand::thread_rng();

        loop {
            let mut v: Vec<i32> = Vec::new();
            for j in 0..dimension {
                v.push(rng.gen_range(lower_bound[j]..=upper_bound[j]));
            }

            if !solutions.contains(&v) {
                return v;
            }
        }
    }

    pub(crate) fn crossover(&self) -> Vec<Arm> {
        let mut crossover_pop: Vec<Arm> = Vec::new();
        let m = self.get_population_size();

        for i in (0..m).step_by(2) {
            if rand::random::<f64>() < self.crossover_rate {
                // Crossover
                let max_dim_index = self.dimension - 1;
                let swap_rv = rand::random::<usize>() % max_dim_index + 1;

                for j in 1..=max_dim_index {
                    if swap_rv == j {
                        let mut cross_vec_1: Vec<i32> = Vec::new();
                        let mut cross_vec_2: Vec<i32> = Vec::new();

                        for h in 0..j {
                            cross_vec_1.push(self.individuals[i].get_action_vector()[h]);
                        }
                        for h in j..=max_dim_index {
                            cross_vec_1.push(self.individuals[i + 1].get_action_vector()[h]);
                        }
                        for h in 0..j {
                            cross_vec_2.push(self.individuals[i + 1].get_action_vector()[h]);
                        }
                        for h in j..=max_dim_index {
                            cross_vec_2.push(self.individuals[i].get_action_vector()[h]);
                        }

                        let new_individual_1 = Arm::new(self.opti_function, cross_vec_1);
                        let new_individual_2 = Arm::new(self.opti_function, cross_vec_2);

                        crossover_pop.push(new_individual_1);
                        crossover_pop.push(new_individual_2);
                    }
                }
            } else {
                // No Crossover
                crossover_pop.push(self.individuals[i].clone());  // Assuming your Arm struct implements the Clone trait
                crossover_pop.push(self.individuals[i + 1].clone());  // Assuming your Arm struct implements the Clone trait
            }
        }

        crossover_pop
    }

    pub(crate) fn mutate(&self, population: Vec<Arm>) -> Vec<Arm> {
        let mut mutated_population: Vec<Arm> = Vec::new();
        let mut seen = HashSet::new();

        for individual in &population {
            let mut new_action_vector = individual.get_action_vector().clone();
            for i in 0..self.dimension {
                if rand::random::<f64>() < self.mutation_rate {
                    let mut new_value = new_action_vector[i] as f64;
                    new_value += Normal::new(0.0, self.mutation_span * (self.upper_bound[i] - self.lower_bound[i]) as f64)
                        .unwrap()
                        .sample(&mut rand::thread_rng());
                    new_value = new_value.max(self.lower_bound[i] as f64);
                    new_value = new_value.min(self.upper_bound[i] as f64);
                    new_action_vector[i] = new_value as i32;
                }
            }
            let new_individual = Arm::new(individual.arm_fn, new_action_vector);  // Assuming you have a constructor like this
            // Only insert new_individual into mutated_population if it hasn't been seen yet
            if seen.insert(new_individual.clone()) {
                mutated_population.push(new_individual);
            }

        }

        mutated_population
    }
}