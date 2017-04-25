from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random
from collections import OrderedDict


class BenchmarkAgent(TrafficLightControl):
    """ represent benchmark agent"""

    def __init__(self, period=4):
        self.period = period
        self.lights = []
        self.last_updated = 0
        self.lightPositions = OrderedDict()

    def build(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def signal(self, t):
        if t - self.last_updated >= self.period:
            for l in self.lights:
                l.switch()
            self.last_updated = t


def run():
    agent = BenchmarkAgent()
    env = Environment(control=agent, grid_size=(8, 6))
    max = env.grid_size[0] * env.grid_size[1] * 6
    for i in range(1, 200):
        pos = random.choice(env.roads.keys())
        heading, obj = env.roads[pos]
        if obj is not None:
            """ find an empty space from start position"""
            for pos, v in env.roads.iteritems():
                if v[1] is None:
                    break
        Car(env, pos)

    simulator = Simulator(env, update_delay=0.2)
    simulator.run()

if __name__ == '__main__':
    run()
