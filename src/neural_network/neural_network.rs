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

        let is_softmax: bool = matches!(self.output.first(), Some(PerceptronKind::Softmax(_)));

        if is_softmax {
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
        }

        for neuron in self.output.iter_mut() {
            neuron.activation_function(&current_inputs);
        }

        self.output
            .iter_mut()
            .map(|neuron| neuron.data().activation)
            .collect()
    }

    pub fn backward_propagation(&mut self, x: &[f64], y: &[f64]) {
        let mut layer_inputs: Vec<Vec<f64>> = Vec::new();
        layer_inputs.push(x.to_vec());

        for layer in &self.hidden {
            let activations: Vec<f64> = layer
                .iter()
                .map(|neuron| neuron.data().activation)
                .collect();
            layer_inputs.push(activations);
        }

        let old_output_weights: Vec<Vec<f64>> = self
            .output
            .iter()
            .map(|neuron| neuron.data().weights.clone())
            .collect();

        let output_inputs = &layer_inputs[self.hidden.len()];

        let mut next_deltas: Vec<f64> = Vec::new();

        for (i, neuron) in self.output.iter_mut().enumerate() {
            let error = neuron.data().activation - y[i];
            let deriv = neuron.derevative();
            let delta = error * deriv;
            next_deltas.push(delta);

            let data = neuron.data_mut();
            for (j, inp) in output_inputs.iter().enumerate() {
                data.weights[j] -= delta * inp * self.learning_rate;
            }
            data.bias -= delta * self.learning_rate;
        }

        let mut old_next_weights = old_output_weights;

        for layer_idx in (0..self.hidden.len()).rev() {
            let current_inputs = &layer_inputs[layer_idx];

            let old_current_weights: Vec<Vec<f64>> = self.hidden[layer_idx]
                .iter()
                .map(|neuron| neuron.data().weights.clone())
                .collect();

            let mut current_deltas: Vec<f64> = Vec::new();

            for (i, neuron) in self.hidden[layer_idx].iter_mut().enumerate() {
                let sum_error: f64 = next_deltas
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| d * old_next_weights[j][i])
                    .sum();

                let deriv = neuron.derevative();
                let delta = sum_error * deriv;
                current_deltas.push(delta);

                let data = neuron.data_mut();
                for (k, inp) in current_inputs.iter().enumerate() {
                    data.weights[k] -= delta * inp * self.learning_rate;
                }
                data.bias -= delta * self.learning_rate;
            }

            next_deltas = current_deltas;
            old_next_weights = old_current_weights;
        }
    }
}
