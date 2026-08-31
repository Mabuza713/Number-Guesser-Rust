pub struct PerceptronData {
    weights: Vec<f64>,
    bias: f64,
    pub activation: f64,
}
pub trait Perceptron {
    fn data(&self) -> &PerceptronData;

    fn weighted_sum(&self, inputs: Vec<f64>) -> f64 {
        let data: &PerceptronData = self.data();

        data.weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + data.bias
    }

    fn activation_function(&mut self, inputs: Vec<f64>);
}

struct ReLUPerceptron {
    pub data: PerceptronData,
}

impl Perceptron for ReLUPerceptron {
    fn data(&self) -> &PerceptronData {
        &self.data
    }

    fn activation_function(&mut self, inputs: Vec<f64>) {
        self.data.activation = self.weighted_sum(inputs).max(0.)
    }
}
