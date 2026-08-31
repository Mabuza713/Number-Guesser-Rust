use crate::{
    image::Image,
    perceptron::{Perceptron, PerceptronKind},
};

struct NeuralNetwork {
    hidden: Vec<Vec<PerceptronKind>>,
    pub output: Vec<PerceptronKind>,
    learning_rate: f64,
}
impl NeuralNetwork {
    pub fn forward_propagation(&mut self, image: Image) -> Vec<f64> {
        let mut current_inputs = image.flatten_matrix().clone();

        for layer in &mut self.hidden {
            for neuron in layer.iter_mut() {
                neuron.activation_function(&current_inputs);
            }

            current_inputs = layer
                .iter()
                .map(|neuron| neuron.data().activation)
                .collect()
        }

        let sum_exp: f64 = self
            .output
            .iter_mut()
            .filter_map(|neuron| neuron.as_softmax_mut())
            .map(|neuron| neuron.compute_exp(&current_inputs))
            .sum();

        self.output
            .iter_mut()
            .filter_map(|neuron| neuron.as_softmax_mut())
            .for_each(|neuron| neuron.set_sum_exp(sum_exp));

        for neuron in self.output.iter_mut() {
            neuron.activation_function(&current_inputs);
        }

        self.output
            .iter()
            .map(|neuron| neuron.data().activation)
            .collect()
    }
}
