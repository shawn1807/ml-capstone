from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralNetwork(object):

    def __init__(self, learning=False, epsilon=1.0, alpha=0.5):
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.hidden_layers = None
        self.output_layer = None
        self.input_layer = None
        
    def set_input_layer(self, input_size):
        self.input_layer = InputLayer(input_size)
    
    def add_hidden_layer(self, number_of_neurons):
        if self.hidden_layers is None:
            self.hidden_layers = Layer(None, number_of_neurons)
        else:
            self.hidden_layers.extend(Layer(None, number_of_neurons))

    def set_output_layer(self, number_of_neurons):
        self.output_layer = Layer(None, number_of_neurons)

    def build(self):
        # connect input, hidden, output layers
        self.input_layer.extend(self.hidden_layers)
        self.hidden_layers.extend(self.output_layer)
        # initial each layers
        layer = self.input_layer
        layer.init()
        while layer.next is not None:
            layer = layer.next
            layer.build()

    def signal(self, x_array):
        output = self.input_layer.forward_feed(np.array(x_array))
        if self.learning:
            self.output_layer.backprop()
        return output


class Layer(object):

    def __init__(self, size):
        self.prior = None
        self.next = None
        self.neurons = []
        self.size = size
        self.built = False

    def extend(self, layer):
        if self.next is None:
            self.next = layer
            layer.prior = self
        else:
            self.next.extend(layer)

    def build(self):
        if not self.built:
            if self.prior is not None:
                for n in range(0, self.size):
                    self.neurons.append(Neuron(len(self.prior.size)))
            self.built = True

    def forward_feed(self, x_in):
        output = np.array([n.evaluate(x_in) for n in self.neurons])
        if self.next is not None:
            return self.next.forward_feed(output)
        else:
            return output

    def backprop(self, loss):
        if self.prior is not None:
            self.prior.backprop()


class InputLayer(Layer):

    def __init__(self, input_size):
        self.input_size = input_size
        self.next = None
        self.built = False

    def build(self):
        if not self.built:
            if self.prior is not None:
                for n in range(0, self.size):
                    self.neurons.append(Neuron(len(self.input_size)))
            self.built = True

    def forward_feed(self, x_in):
        return self.next.forward_feed(x_in)

            
class Neuron(object):

    def __init__(self, n_in, bias):
        self.weights = np.random.randn(n_in)
        self.bias = bias

    def evaluate(self, x):
        axon = np.sum(self.weights.dot(x)) + self.bias
        return sigmoid(axon)


class QLearningNetworkAgent(TrafficLightControl):
    """ represent Q learning network agent"""

    def __init__(self, period=2, learning=False, epsilon=1.0, alpha=0.5):
        super(QLearningNetworkAgent, self).__init__()
        self.period = period
        self.lights = []
        self.lightPositions = OrderedDict()
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.neural_network = None

    def build_light(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def setup(self):
        self.neural_network = NeuralNetwork(epsilon=self.epsilon, alpha=self.alpha)
        self.neural_network.set_input_layer(len(self.env.roads))
        self.neural_network.add_hidden_layer(len(self.lights))
        self.neural_network.set_output_layer(len(self.lights))

    def signal(self):
        if self.env.t % self.period == 0:
            x = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            x = np.append(x, [1])
            output = self.neural_network.signal(x)
            print output
            if self.learning:
                for l in self.lights:
                    l.switch()
            else:
                pass

    def after_signal(self):
        if self.learning:
            pass

    def reset(self):
        pass

    def allow(self, position, heading):
        return super(QLearningNetworkAgent, self).allow(position, heading)


def run():
    trials = 100
    cars = [10, 50, 150, 200, 250, 300]
    period = 0.5
    agent = QLearningNetworkAgent(learning=True)
    env = Environment(control=agent, grid_size=(8, 4))
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
    run()
