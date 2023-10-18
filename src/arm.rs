pub(crate) struct Arm {
    action_vector: Vec<i32>,
    reward: f64,
    num_pulls: i32,
    pub(crate) arm_fn: fn(Vec<i32>) -> f64,
}

impl Arm {
    pub(crate) fn new(arm_fn: fn(Vec<i32>) -> f64, action_vector: Vec<i32>) -> Arm {
        Arm {
            reward: 0.0,
            num_pulls: 0,
            arm_fn,
            action_vector,
        }
    }

    pub(crate) fn pull(&mut self) -> f64 {
        let g = (self.arm_fn)(self.action_vector.clone());

        self.reward += g;
        self.num_pulls += 1;

        return g;
    }

    pub(crate) fn get_num_pulls(&self) -> i32 {
        return self.num_pulls;
    }

    pub(crate) fn get_function_value(&self) -> f64 {
        return (self.arm_fn)(self.action_vector.clone());
    }

    pub(crate) fn get_action_vector(&self) -> Vec<i32> {
        return self.action_vector.clone();
    }

    pub(crate) fn get_mean_reward(&self) -> f64 {
        return self.reward / self.num_pulls as f64;
    }

}

impl Clone for Arm {
    fn clone(&self) -> Self {
        Self {
            action_vector: self.action_vector.clone(),
            reward: self.reward,
            num_pulls: self.num_pulls,
            arm_fn: self.arm_fn,
        }
    }
}

use std::hash::{Hash, Hasher};

impl PartialEq for Arm {
    fn eq(&self, other: &Self) -> bool {
        self.action_vector == other.action_vector
    }
}

impl Eq for Arm {}

impl Hash for Arm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action_vector.hash(state);
    }
}
