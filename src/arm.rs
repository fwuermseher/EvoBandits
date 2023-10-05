struct Arm {
    action_vector: Vec<i32>,
    reward: f64,
    num_pulls: i32,
    arm_fn: fn(Vec<i32>) -> f64,
}

impl Arm {
    fn new(arm_fn: fn(Vec<i32>) -> f64, action_vector: Vec<i32>) -> Arm {
        Arm {
            reward: 0.0,
            num_pulls: 0,
            arm_fn,
            action_vector,
        }
    }

    fn pull(&mut self) -> f64 {
        let g = (self.arm_fn)(self.action_vector.clone());

        self.reward += g;
        self.num_pulls += 1;

        return g;
    }

    fn get_reward(&self) -> f64 {
        return self.reward;
    }

    fn get_num_pulls(&self) -> i32 {
        return self.num_pulls;
    }

    fn get_function_value(&self) -> f64 {
        return (self.arm_fn)(self.action_vector.clone());
    }

}