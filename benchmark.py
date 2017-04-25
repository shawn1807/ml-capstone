from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random
from collections import OrderedDict
import csv

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

    def reset(self):
        self.lights = []
        self.last_updated = 0
        self.lightPositions = OrderedDict()

def run():
    trials = 10
    cars = [10, 100, 200, 300]
    period = 10
    agent = BenchmarkAgent()
    env = Environment(control=agent, grid_size=(8, 6))
    log_file = open("benchmark", 'wb')
    log_writer = csv.DictWriter(log_file,fieldnames=["trial","cars","total_stall","average","score"])
    log_writer.writeheader()
    simulator = Simulator(env, update_delay=0.1,logger=log_writer)
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
