from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict
from ann import NeuralNetwork

from ann import cross_entropy


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


class LearningAgent(TrafficLightControl):
    """ neural network agent"""

    def __init__(self, period=2, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__()
        self.period = period
        self.lights = []
        self.lightPositions = OrderedDict()
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.model = None
        self.input_x = None
        self.learning_count = 0

    def build_light(self):
        tl = NeuronTrafficLight()
        self.lights.append(tl)
        return tl

    def stop_learning(self):
        self.learning = False

    def setup(self):
        self.model = NeuralNetwork(alpha=self.alpha)
        self.model.set_input_layer(len(self.env.roads))
        self.model.add_hidden_layer(len(self.env.roads), activation="linear")
        self.model.set_output_layer(len(self.lights), activation="sigmoid")
        self.model.build()
        self.model
        for i in range(0, len(self.lights)):
            self.lights[i].neuron = self.model.output_layer.neurons[i]

    def signal(self):
        if self.env.t % self.period == 0:
            self.input_x = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            self.model.signal(self.input_x)
            for l in self.lights:
                l.switch()

    def after_signal(self):
        if self.env.t % self.period == 0 and self.learning:
            self.learning_count += 1
            expected = []
            for l in self.lights:
                best = 0.5
                if l.ns_vote > l.ew_vote:
                    best = 1
                elif l.ns_vote < l.ew_vote:
                    best = 0

                expected.append(best)
                l.ns_vote = 0
                l.ew_vote = 0
            self.model.back_propagate(expected)
            self.model.update()

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
        return super(LearningAgent, self).allow(position,heading)


def run():
    trials = 2
    cars = [50, 100]
    period = 20
    agent = LearningAgent(learning=True, alpha=0.15)
    env = Environment(control=agent, grid_size=(8, 4))
    simulator = Simulator(env, update_delay=0.1, filename="ann_training_32L.csv")
    simulator.title = "Training ANN Agent"
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
    #testing
    agent.stop_learning()
    simulator.title = "ANN Agent"
    simulator.reset_logger_file("ann_testing_32L.csv")
    period = 5
    for ncar in cars:
        env.reset()
        for i in range(1, ncar + 1):
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

