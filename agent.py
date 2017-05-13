from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralNetwork(object):

    def __init__(self, learning=False, epsilon=1.0, alpha=0.7):
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.hidden_layers = None
        self.output_layer = None
        self.input_layer = None

    def set_input_layer(self, input_size):
        self.input_layer = InputLayer(self, input_size)
    
    def add_hidden_layer(self, number_of_neurons, activation= "sigmoid"):
        if self.hidden_layers is None:
            self.hidden_layers = Layer(self, "Hidden Layer", number_of_neurons, activation)
        else:
            self.hidden_layers.extend(Layer(self, "Hidden Layer", number_of_neurons, activation))

    def set_output_layer(self, number_of_neurons, activation = "sigmoid"):
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
        output = self.input_layer.forward_feed(np.array(x_array))
        return output

    def back_propagate(self, errors):
        self.output_layer.back_propagate(errors)
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
                        neuron = SigmoidNeuron(self.prior.size, np.random.random())
                    elif self.activation == "relu":
                        neuron = ReluNeuron(self.prior.size, np.random.random())
                    elif self.activation == "standard":
                        neuron = Neuron(self.prior.size, np.random.random())
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
        for i in range(0, len(errors)):
            self.neurons[i].propagate_error(errors[i])
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

    def __str__(self):
        return "%s(units:%s, %s)" % (self.name, self.size, [str(n) for n in self.neurons])


class Neuron(object):

    def __init__(self, n_in, bias):
        self.weights = np.random.randn(n_in)
        self.bias = bias
        self.input = None
        self.output = None
        self.delta = np.zeros(n_in)
        self.batch = 0
        self.error_term = 0

    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = axon
        return self.output

    def propagate_error(self, err):
        self.error_term = err
        self.delta += self.input * self.error_term
        self.batch += 1

    def update(self, learning_rate):
        self.weights += learning_rate * self.delta / self.batch

    def __str__(self):
        return "Neuron(%s, %s)" % (self.weights, self.bias)


class SigmoidNeuron(Neuron):
    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = sigmoid(axon)
        return self.output

    def propagate_error(self, err):
        self.error_term = err * (1-self.output) * self.output
        self.delta += self.input * self.error_term
        self.batch += 1


class ReluNeuron(Neuron):

    def activate(self, x):
        self.input = x
        axon = np.sum(self.weights.dot(x)) + self.bias
        self.output = max(0,axon)
        return self.output

    def propagate_error(self, err):
        self.error_term = err if err > 0 else 0
        self.delta += self.input * self.error_term
        self.batch += 1

import matplotlib.pyplot as plt
import plotly.plotly as py


def test():
    network = NeuralNetwork()
    network.set_input_layer(2)
    network.add_hidden_layer(4)
    network.set_output_layer(1)
    network.build()
    errors = []
    line = plt.figure()
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(0,2000):
        output = network.signal(np.array([1, 0]))
        error = 1- output
        plt.plot(i, error, "o", color=(0,0,0))
        errors.append(error)
        network.back_propagate([error])
        output = network.signal(np.array([0, 1]))
        error = 1 - output
        plt.plot(i, error, "o", color=(1,0,0))
        errors.append(error)
        network.back_propagate([error])

        output = network.signal(np.array([1, 1]))
        error = 0 - output
        plt.plot(i, error, "o", color=(0, 1, 0))
        errors.append(error)
        network.back_propagate([error])
        output = network.signal(np.array([0, 0]))
        error = 0 - output
        plt.plot(i, error, "o", color=(1, 1, 0))
        errors.append(error)
        network.back_propagate([error])
    print network.signal(np.array([1, 0]))
    print network.signal(np.array([0, 1]))
    print network.signal(np.array([1, 1]))
    print network.signal(np.array([0, 0]))
    plt.show()



class NeuronTrafficLight(TrafficLight):

    def __init__(self, open_way=None):
        super(NeuronTrafficLight, self).__init__(open_way)
        self.neuron = None
        self.ns_vote = 0
        self.ew_vote = 0

    def switch(self):
        """if neuron output is greater than 0.5 then opens North-South way"""
        if self.neuron.output < 0.4:
            self.open_way = self.EW
        elif self.neuron.output > 0.6:
            self.open_way = self.NS
        else:
            self.open_way = random.choice([self.EW, self.NS])


class NeuralNetworkAgent(TrafficLightControl):
    """ represent Q learning network agent"""

    def __init__(self, period=2, learning=False, epsilon=1.0, alpha=0.5):
        super(NeuralNetworkAgent, self).__init__()
        self.period = period
        self.lights = []
        self.lightPositions = OrderedDict()
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.neural_network = None
        self.input_x = None
        self.learning_count = 0

    def build_light(self):
        tl = NeuronTrafficLight()
        self.lights.append(tl)
        return tl

    def setup(self):
        self.neural_network = NeuralNetwork(epsilon=self.epsilon, alpha=self.alpha)
        self.neural_network.set_input_layer(len(self.env.roads))
        #self.neural_network.add_hidden_layer(len(self.env.roads), activation="relu")
        self.neural_network.add_hidden_layer(len(self.env.roads), activation="sigmoid")
        self.neural_network.set_output_layer(len(self.lights), activation="sigmoid")
        self.neural_network.build()
        for i in range(0, len(self.lights)):
            self.lights[i].neuron = self.neural_network.output_layer.neurons[i]

    def signal(self):
        if self.env.t % self.period == 0:
            self.input_x = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            print "output", self.neural_network.signal(self.input_x)
            for l in self.lights:
                l.switch()

    def after_signal(self):
        if self.env.t % self.period == 0 and self.learning:
            self.learning_count += 1
            errors = []
            for l in self.lights:
                error = 0
                if l.ns_vote > l.ew_vote:
                    error = 1 - l.neuron.output
                elif l.ns_vote < l.ew_vote:
                    error = 0 - l.neuron.output
                else:
                    error = 0.5 - l.neuron.output
                errors.append(error)
                print "NS vote:", l.ns_vote, "EW vote", l.ew_vote
                l.ns_vote = 0
                l.ew_vote = 0
            print self.learning_count, ":Errors", errors
            self.neural_network.back_propagate(errors)

    def reset(self):
        pass

    def allow(self, position, heading):
        """already in intersections"""
        if self.lightPositions.has_key(position):
            return True
        """check next position if it is in the intersections"""
        pos = (position[0] + heading[0]*2, position[1] + heading[1]*2)
        if self.lightPositions.has_key(pos):
            light = self.lightPositions[pos]
            if light.get_open_way() == TrafficLight.NS:
                allowPassing = heading == (0, 1) or heading == (0, -1)
                if allowPassing:
                    light.ns_vote += 1
                else:
                    light.ew_vote += 1
            else:
                allowPassing = heading == (1, 0) or heading == (-1, 0)
                if allowPassing:
                    light.ew_vote += 1
                else:
                    light.ns_vote += 1
        return super(NeuralNetworkAgent,self).allow(position,heading)


def run():
    trials = 5
    cars = [1]
    period = 10
    agent = NeuralNetworkAgent(learning=True, alpha=0.9)
    env = Environment(control=agent, grid_size=(2, 2))
    simulator = Simulator(env, update_delay=0.1, filename="agent.csv")
    simulator.title = "Training Learning Agent"
    for t in range(1, trials+1):
        for ncar in cars:
            env.reset()
            for i in range(1, ncar+1):
                pos = random.choice(env.roads.keys())
                heading, obj = env.roads[pos]
                if obj is not None:
                    """ find an empty space from start position"""
                    for pos, v in env.roads.iteritems():
                        if v[1] is None:
                            break
                Car(env, pos)
            simulator.run(t, period)
    simulator.quit()


if __name__ == '__main__':
    #run()
    test()
