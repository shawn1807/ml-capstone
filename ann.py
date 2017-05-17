import logging
import numpy as np
import matplotlib.pyplot as plt
import math
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)


def test():
    network = NeuralNetwork(batch_size=1,alpha=0.8,learning=True)
    network.set_input_layer(2)
    network.add_hidden_layer(20)
    network.set_output_layer(1)
    network.build()
    errors = []
    line = plt.figure()

    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(0,1000):
        output = network.signal(np.array([1, 0]))
        error = 1- output
        plt.plot(i, error, "x", color=(0,0,0))
        errors.append(error)
        network.back_propagate([1])
        output = network.signal(np.array([0, 1]))
        error = 1 - output
        plt.plot(i, error, "o", color=(1,0,0))
        errors.append(error)
        network.back_propagate([1])

        output = network.signal(np.array([1, 1]))
        error = 1 - output
        plt.plot(i, error, "o", color=(0, 1, 0))
        errors.append(error)
        network.back_propagate([0])
        output = network.signal(np.array([0, 0]))
        error = 0 - output
        plt.plot(i, error, "o", color=(1, 1, 0))
        errors.append(error)
        network.back_propagate([0])
    print network.signal(np.array([1, 0]))
    print network.signal(np.array([0, 1]))
    print network.signal(np.array([1, 1]))
    print network.signal(np.array([0, 0]))
    plt.show()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cross_entropy(a, y):
    return (- math.log1p(a)*y + (1-y) * math.log1p(1-a))

def quadratic(a, y):
    return a - y


class NeuralNetwork(object):

    def __init__(self, learning=False, epsilon=1.0, alpha=0.7, batch_size = 1,cost_func=quadratic):
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.hidden_layers = None
        self.output_layer = None
        self.input_layer = None
        self.output = None
        self.batch_size = batch_size
        self.count = 0
        self.cost_func = cost_func

    def set_input_layer(self, input_size):
        self.input_layer = InputLayer(self, input_size)

    def add_hidden_layer(self, number_of_neurons, activation="sigmoid"):
        if self.hidden_layers is None:
            self.hidden_layers = Layer(self, "Hidden Layer", number_of_neurons, activation)
        else:
            self.hidden_layers.extend(Layer(self, "Hidden Layer", number_of_neurons, activation))

    def set_output_layer(self, number_of_neurons, activation="sigmoid"):
        self.output_layer = OutputLayer(self, "Output Layer", number_of_neurons, activation)

    def build(self):
        # connect input, hidden, output layers
        self.input_layer.extend(self.hidden_layers)
        self.hidden_layers.extend(self.output_layer)
        # initial each layers
        layer = self.input_layer
        layer.build()
        while layer.next is not None:
            layer = layer.next
            layer.build()
        logger.debug("Neural network built [%s]" % self)

    def signal(self, x_array):
        self.output = self.input_layer.forward_feed(np.array(x_array))
        return self.output

    def back_propagate(self, y):
        self.output_layer.back_propagate([self.cost_func(a, yi) for a, yi in zip(self.output, y)])
        self.count += 1
        if self.count % self.batch_size == 0:
            for l in self.hidden_layers:
                l.update()

    def __str__(self):
        return "(learning rate:%s) %s" % (self.alpha, [str(l) for l in self.input_layer])


class Layer(object):
    def __init__(self, network, name, size, activation):
        self.network = network
        self.prior = None
        self.name = name
        self.next = None
        self.neurons = []
        self.size = size
        self.built = False
        self.index = 0
        self.activation = activation

    def extend(self, layer):
        if self.next is None:
            self.next = layer
            layer.prior = self
            layer.index = self.index + 1
        else:
            self.next.extend(layer)

    def build(self):
        if not self.built:
            if self.prior is not None:
                for n in range(0, self.size):
                    if self.activation == "sigmoid":
                        neuron = SigmoidNeuron(self.prior.size)
                    elif self.activation == "relu":
                        neuron = ReluNeuron(self.prior.size)
                    elif self.activation == "standard":
                        neuron = Neuron(self.prior.size)
                    else:
                        raise ValueError("activation function [%s] not supported" % self.activation)
                    self.neurons.append(neuron)
            self.built = True

    def forward_feed(self, x_in):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logger.debug("Input: %s" % x_in)
            logger.debug("Neurons: %s" % [str(n) for n in self.neurons])
        output = np.array([n.activate(x_in) for n in self.neurons])
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logger.debug("Output: %s" % output)
        if self.next is not None:
            return self.next.forward_feed(output)
        else:
            return output

    def back_propagate(self, errors):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logger.debug("back_propagate errors: %s" % errors)
            logger.debug("Neurons: %s" % [str(n) for n in self.neurons])
        # propagate error to each neuron
        for n, e in zip(self.neurons, errors):
            n.propagate_error(e)
        # calculate prior layer error
        if self.prior is not None:
            prior_errors = np.sum(np.array([n.weights * n.error_term for n in self.neurons]).transpose(), axis=1)
            self.prior.back_propagate(prior_errors)

    def update(self):
        for n in self.neurons:
            n.update(self.network.alpha)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logger.debug("Updated Neurons: %s" % [str(n) for n in self.neurons])

    def __str__(self):
        return "%s:%s(units:%s, %s)" % (self.name, self.index, self.size, [str(n) for n in self.neurons])

    def __iter__(self):
        layer = self
        while layer is not None:
            yield layer
            layer = layer.next


class InputLayer(Layer):
    def __init__(self, network, input_size):
        self.network = network
        self.size = input_size
        self.next = None
        self.built = False
        self.name = "Input Layer"
        self.index = 0

    def build(self):
        pass

    def forward_feed(self, x_in):
        """Input layer doesn't do anything, pass input to next layer"""
        return self.next.forward_feed(x_in)

    def back_propagate(self, errors):
        pass

    def __str__(self):
        return "%s(units:%s)" % (self.name, self.size)


class OutputLayer(Layer):
    def __str__(self, ):
        return "%s(units:%s, %s)" % (self.name, self.size, [str(n) for n in self.neurons])


class Neuron(object):

    def __init__(self, n_in):
        self.weights = np.random.randn(n_in)/np.sqrt(n_in)
        self.bias = np.random.rand()
        self.input = None
        self.output = None
        self.delta = np.zeros(n_in)
        self.batch = 0
        self.error_term = 0
        self.errors = 0

    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = axon
        return self.output

    def propagate_error(self, err):
        self.errors += err
        self.error_term = err
        self.delta += self.input * self.error_term
        self.batch += 1

    def update(self, learning_rate):
        self.weights -= learning_rate * self.delta / self.batch
        self.bias -= self.errors / self.batch
        self.delta = 0.0
        self.batch = 0
        self.errors = 0.0

    def __str__(self):
        return "Neuron(%s, %s)" % (self.weights, self.bias)


class SigmoidNeuron(Neuron):

    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = sigmoid(axon)
        return self.output

    def propagate_error(self, err):
        self.errors += err
        self.error_term = err * (1 - self.output) * self.output
        self.delta += self.input * self.error_term
        self.batch += 1


class ReluNeuron(Neuron):

    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = max(0, axon)
        return self.output

    def propagate_error(self, err):
        self.errors += err
        self.error_term = err if err > 0 else 0
        self.delta += self.input * self.error_term
        self.batch += 1

if __name__ == '__main__':
    test()