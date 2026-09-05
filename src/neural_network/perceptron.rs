use std::{ptr::read, result};

pub struct PerceptronData {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: f64,
}
pub trait Perceptron {
    fn data(&self) -> &PerceptronData;

    fn data_mut(&mut self) -> &mut PerceptronData;

    fn weighted_sum(&self, inputs: &Vec<f64>) -> f64 {
        let data: &PerceptronData = self.data();

        data.weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + data.bias
    }

    fn activation_function(&mut self, inputs: &Vec<f64>);

    fn derevative(&mut self) -> f64;
}

struct ReLUPerceptron {
    pub data: PerceptronData,
}

impl Perceptron for ReLUPerceptron {
    fn data(&self) -> &PerceptronData {
        &self.data
    }
    fn data_mut(&mut self) -> &mut PerceptronData {
        &mut self.data
    }

    fn activation_function(&mut self, inputs: &Vec<f64>) {
        self.data.activation = self.weighted_sum(inputs).max(0.)
    }

    fn derevative(&mut self) -> f64 {
        if self.data.activation > 0. {
            return 1.;
        } else {
            return 0.;
        }
    }
}

pub struct Softmax {
    pub data: PerceptronData,
    pub exp_value: f64,
    pub sum_exp: f64,
}
impl Perceptron for Softmax {
    fn data(&self) -> &PerceptronData {
        &self.data
    }
    fn data_mut(&mut self) -> &mut PerceptronData {
        &mut self.data
    }

    fn activation_function(&mut self, inputs: &Vec<f64>) {
        if self.exp_value == 0. {
            self.compute_exp(inputs);
        }
        self.data.activation = self.data.activation / self.sum_exp
    }

    fn derevative(&mut self) -> f64 {
        self.data.activation * (1. - self.data.activation)
    }
}
impl Softmax {
    pub fn compute_exp(&mut self, inputs: &Vec<f64>) -> f64 {
        let result: f64 = self.weighted_sum(inputs);
        self.exp_value = result.exp();
        self.exp_value
    }

    pub fn set_sum_exp(&mut self, value: f64) {
        self.sum_exp = value
    }
}

pub enum PerceptronKind {
    ReLU(ReLUPerceptron),
    Softmax(Softmax),
}

impl Perceptron for PerceptronKind {
    fn data(&self) -> &PerceptronData {
        match self {
            PerceptronKind::ReLU(p) => p.data(),
            PerceptronKind::Softmax(p) => p.data(),
        }
    }

    fn data_mut(&mut self) -> &mut PerceptronData {
        match self {
            PerceptronKind::ReLU(p) => p.data_mut(),
            PerceptronKind::Softmax(p) => p.data_mut(),
        }
    }

    fn activation_function(&mut self, inputs: &Vec<f64>) {
        match self {
            PerceptronKind::ReLU(p) => p.activation_function(inputs),
            PerceptronKind::Softmax(p) => p.activation_function(inputs),
        }
    }

    fn derevative(&mut self) -> f64 {
        match self {
            PerceptronKind::ReLU(p) => p.derevative(),
            PerceptronKind::Softmax(p) => p.derevative(),
        }
    }
}

impl PerceptronKind {
    pub fn as_softmax_mut(&mut self) -> Option<&mut Softmax> {
        match self {
            PerceptronKind::Softmax(s) => Some(s),
            _ => None,
        }
    }
}
