from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict


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
        self.weights_h1 = None
        self.weights_h2 = None

    def build_light(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def setup(self):
        self.weights_h1 = np.random.randn(len(self.env.roads)+1)
        self.weights_h2 = np.random.randn(len(self.lights))

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
