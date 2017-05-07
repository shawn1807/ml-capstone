from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict


class NeuralNetwork(object):

    def __init__(self, epsilon=1.0, alpha=0.5):
        self.epsilon = epsilon
        self.alpha = alpha

    def add_hidden_layer(self, number_of_neurons):
        if self.first_layer is None:
            self.first_layer = Layer(None, number_of_neurons)
        else:
            self.first_layer.extend(number_of_neurons)

    def react(self, x_array):
        return self.first_layer.forward(np.array(x_array))


class Layer(object):

    def __init__(self, prior, number_of_neurons):
        self.prior = prior
        self.next = None
        self.neurons = None
        if self.prior is not None:
            self.prior.next = self
            for n in number_of_neurons:
                self.neurons.append(Neuron(len(self.prior.neurons)))

    def extend(self, number_of_neurons):
        if self.next is None:
            self.next = Layer(self, number_of_neurons)
        else:
            self.next.attach(number_of_neurons)

    def forward(self, x_in):
        output = np.array([n.evaluate(x_in) for n in self.neurons])
        if self.next is not None:
            return self.next.forward(output)
        else:
            return output

    def backward(self, loss):
        pass


class Neuron(object):

    def __init__(self, n_in, bias):
        self.weights = np.random.randn(n_in)
        self.bias = bias

    def evaluate(self, x):
        axon = np.sum(self.weights.dot(x)) + self.bias
        return axon


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
        self.neural_network.add_hidden_layer(len(self.lights))


    def signal(self):
        if self.env.t % self.period == 0:
            x = np.array([0 if o is None else 1 for p, o in self.env.roads.values() ])
            x = np.append(x, [1])
            print self.weights_h1.dot(x)
            if self.learning:
                for l in self.lights:
                    l.switch()
            else:
                pass

    def after_signal(self):
        print "after_signal"

    def reset(self):
        pass

    def allow(self, position, heading):
        allowed = super(QLearningNetworkAgent, self).allow(position, heading)
        return allowed


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
