use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub(crate) struct Arm {
    action_vector: Vec<i32>,
    reward: f64,
    num_pulls: i32,
    pub(crate) arm_fn: fn(&[i32]) -> f64,
}

impl Arm {
    pub(crate) fn new(arm_fn: fn(&[i32]) -> f64, action_vector: &[i32]) -> Self {
        Self {
            reward: 0.0,
            num_pulls: 0,
            arm_fn,
            action_vector: action_vector.to_vec(),
        }
    }

    pub(crate) fn pull(&mut self) -> f64 {
        let g = (self.arm_fn)(&self.action_vector);

        self.reward += g;
        self.num_pulls += 1;

        g
    }

    pub(crate) fn get_num_pulls(&self) -> i32 {
        self.num_pulls
    }

    pub(crate) fn get_function_value(&self) -> f64 {
        (self.arm_fn)(&self.action_vector)
    }

    pub(crate) fn get_action_vector(&self) -> &[i32] {
        &self.action_vector
    }

    pub(crate) fn get_mean_reward(&self) -> f64 {
        self.reward / self.num_pulls as f64
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

#[cfg(test)]
mod tests {
    use super::*;

    // Mock optimization function for testing
    fn mock_opti_function(_vec: &[i32]) -> f64 {
        5.0
    }

    #[test]
    fn test_arm_new() {
        let arm = Arm::new(mock_opti_function, &vec![1, 2]);
        assert_eq!(arm.get_num_pulls(), 0);
        assert_eq!(arm.get_function_value(), 5.0);
    }

    #[test]
    fn test_arm_pull() {
        let mut arm = Arm::new(mock_opti_function, &vec![1, 2]);
        let reward = arm.pull();

        assert_eq!(reward, 5.0);
        assert_eq!(arm.get_num_pulls(), 1);
        assert_eq!(arm.get_mean_reward(), 5.0);
    }

    #[test]
    fn test_arm_pull_multiple() {
        let mut arm = Arm::new(mock_opti_function, &vec![1, 2]);
        arm.pull();
        arm.pull();

        assert_eq!(arm.get_num_pulls(), 2);
        assert_eq!(arm.get_mean_reward(), 5.0);  // Since reward is always 5.0
    }

    #[test]
    fn test_arm_clone() {
        let arm = Arm::new(mock_opti_function, &vec![1, 2]);
        let cloned_arm = arm.clone();

        assert_eq!(arm.get_num_pulls(), cloned_arm.get_num_pulls());
        assert_eq!(arm.get_function_value(), cloned_arm.get_function_value());
        assert_eq!(arm.get_action_vector(), cloned_arm.get_action_vector());
    }

    #[test]
    fn test_arm_equality() {
        let arm1 = Arm::new(mock_opti_function, &vec![1, 2]);
        let arm2 = Arm::new(mock_opti_function, &vec![1, 2]);
        let arm3 = Arm::new(mock_opti_function, &vec![2, 1]);

        assert_eq!(arm1, arm2);
        assert_ne!(arm1, arm3);
    }
}
