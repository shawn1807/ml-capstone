from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict


class QLearningNetworkAgent(TrafficLightControl):
    """ represent benchmark agent"""

    def __init__(self, period=2, learning=False, epsilon=1.0, alpha=0.5):
        super(QLearningNetworkAgent, self).__init__()
        self.period = period
        self.lights = []
        self.last_updated = 0
        self.lightPositions = OrderedDict()
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha

    def build(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def signal(self):
        if self.env.t - self.last_updated >= self.period:
            self.last_updated = self.env.t
            x = np.array(input)
            if self.learning:
                for l in self.lights:
                    l.switch()
            else:
                pass

    def after_signal(self):
        print "after_signal"

    def reset(self):
        self.last_updated = 0


def run():
    trials = 100
    cars = [1, 5, 150, 200, 250, 300]
    period = 0.5
    agent = QLearningNetworkAgent(learning=True)
    env = Environment(control=agent, grid_size=(8, 6))
    simulator = Simulator(env, update_delay=0.1, filename="agent.csv")
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
