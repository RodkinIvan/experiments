import numpy as np
from math import sqrt


def act_function_derivative_np(states: np.ndarray):
    return np.ones(states.size)-states*states


def horizontal(vector: np.ndarray) -> np.ndarray:
    return vector.reshape(1, vector.size)


def vertical(vector: np.ndarray) -> np.ndarray:
    return vector.reshape(vector.size, 1)


def clip(vector: np.ndarray, max_len: float) -> np.ndarray:
    len_sq = (vector**2).sum()
    if len_sq > max_len**2:
        return (max_len/sqrt(len_sq))*vector
    else:
        return vector


class MLP:
    def __init__(self, layers_sizes: list):
        layers_num = len(layers_sizes)

        self.layers_num = layers_num

        self.layers = [np.zeros(1)]*layers_num
        self.inputs = [np.zeros(1)]*layers_num

        self.weights = [np.zeros(1)] * (layers_num - 1)
        self.bias = [np.zeros(1)] * (layers_num-1)

        self.error_matrices = []
        for layer in range(layers_num):
            self.layers[layer] = np.ndarray((layers_sizes[layer], 1))
            if layer != layers_num - 1:
                self.weights[layer] = np.random.normal(0, 0.1, (layers_sizes[layer + 1], layers_sizes[layer]))
                self.bias[layer] = np.random.normal(0, 0.1, (layers_sizes[layer+1]))

                self.error_matrices.append(np.random.normal(0, 0.75, (layers_sizes[layer], layers_sizes[layer+1])))

    def common_forward(self, x: np.ndarray):
        x = x.flatten()
        self.layers[0] = x
        for layer in range(self.layers_num - 1):
            self.layers[layer+1] = np.tanh(self.weights[layer] @ self.layers[layer] + self.bias[layer])
        return self.layers[self.layers_num-1]

    def lra_forward(self, x: np.ndarray):
        x = x.flatten()
        self.layers[0] = x
        for layer in range(self.layers_num - 1):
            input_h = self.weights[layer] @ self.layers[layer] + self.bias[layer]
            self.inputs[layer+1] = input_h
            self.layers[layer+1] = np.tanh(input_h)
        return self.layers[self.layers_num-1]

    def forward(self, optimizer: str, inputs):
        if optimizer[:3] == 'lra':
            return self.lra_forward(inputs)
        else:
            return self.common_forward(inputs)

    def backprop(self, targets: np.ndarray, spd: float = 0.003):

        layer = self.layers_num - 1

        delta = self.layers[layer] - targets

        while layer >= 1:
            input_grad = delta * act_function_derivative_np(self.layers[layer])
            self.weights[layer-1] -= spd*vertical(input_grad) @ horizontal(self.layers[layer-1])
            self.bias[layer-1] -= spd*input_grad
            if layer != 1:
                delta = (self.weights[layer-1].transpose() @ vertical(input_grad)).flatten()
            layer -= 1

    def target_backprop(self, targets: np.ndarray, spd: float = 0.003):
        layers_targets = self.layers.copy()

        layer = self.layers_num - 1
        layers_targets[layer] = targets

        while layer >= 1:
            delta = self.layers[layer] - layers_targets[layer]
            input_grad = delta * act_function_derivative_np(self.layers[layer])
            self.weights[layer-1] -= spd*vertical(input_grad) @ horizontal(self.layers[layer-1])
            self.bias[layer-1] -= spd*input_grad
            if layer != 1:
                layers_targets[layer-1] -= (self.weights[layer-1].transpose() @ vertical(input_grad)).flatten()
            layer -= 1

    def lra_step(self, targets: np.ndarray, spd: float = 0.003, fdbk: bool = False, iterations_num: int = 10, clipping: tuple = None):
        layers_targets = self.layers.copy()

        layer = self.layers_num - 1
        layers_targets[layer] = targets

        while layer >= 1:
            delta = self.layers[layer] - layers_targets[layer]
            input_grad = delta * act_function_derivative_np(self.layers[layer])
            if clipping is not None:
                self.weights[layer-1] -= clip(spd*vertical(input_grad) @ horizontal(self.layers[layer-1]), clipping[0])
                self.bias[layer-1] -= clip(spd*input_grad, clipping[0])
            else:
                self.weights[layer-1] -= spd*vertical(input_grad) @ horizontal(self.layers[layer-1])
                self.bias[layer-1] -= spd*input_grad

            prev_inputs = self.inputs[layer-1]
            prev_activations = 0
            for k in range(iterations_num):
                if not fdbk:
                    prev_layer_der = act_function_derivative_np(self.layers[layer-1])
                    delta_h = (self.weights[layer-1].transpose() @  vertical(input_grad)).flatten() * prev_layer_der
                else:
                    delta_h = self.error_matrices[layer-1] @ input_grad
                if clipping is not None:
                    prev_inputs -= clip(delta_h, clipping[1])
                else:
                    prev_inputs -= delta_h
                prev_activations = np.tanh(prev_inputs)
                self.inputs[layer] = self.weights[layer-1] @ prev_activations + self.bias[layer-1]
                self.layers[layer] = np.tanh(self.inputs[layer])
                delta = self.layers[layer] - layers_targets[layer]
                input_grad = delta * act_function_derivative_np(self.layers[layer])
            layers_targets[layer-1] = prev_activations
            layer -= 1

    def FA_step(self, targets: np.ndarray, spd = 0.003):
        layer = self.layers_num - 1

        delta = self.layers[layer] - targets

        while layer >= 1:
            input_grad = delta * act_function_derivative_np(self.layers[layer])
            self.weights[layer-1] -= spd*vertical(input_grad) @ horizontal(self.layers[layer-1])
            self.bias[layer-1] -= spd*input_grad
            if layer != 1:
                delta = self.error_matrices[layer-1] @ delta
            layer -= 1

    def optim_step(self, optimizer, targets: np.ndarray, **kwargs):
        if optimizer == 'lra_diff':
            self.lra_step(targets, **kwargs)
        elif optimizer == 'lra_fdbk':
            self.lra_step(targets, fdbk=True, iterations_num=1, **kwargs)
        elif optimizer == 'bp':
            self.backprop(targets, **kwargs)
        elif optimizer == 'FA':
            self.FA_step(targets, **kwargs)
        else:
            raise 'choose optimizer'
