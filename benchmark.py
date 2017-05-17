from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random
from collections import OrderedDict


class BenchmarkAgent(TrafficLightControl):
    """ represent benchmark agent"""

    def __init__(self, period=2):
        super(BenchmarkAgent, self).__init__()
        self.period = period
        self.lights = []
        self.last_updated = 0
        self.lightPositions = OrderedDict()

    def build_light(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def signal(self):
        if self.env.t - self.last_updated >= self.period:
            for l in self.lights:
                l.switch()
            self.last_updated = self.env.t

    def reset(self):
        self.last_updated = 0


def run():
    trials = 100
    cars = [3]
    period = 5
    agent = BenchmarkAgent(period=4)
    env = Environment(control=agent, grid_size=(2, 2))
    simulator = Simulator(env, update_delay=0.1, title="Benchmark", filename="benchmark.csv")
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
    #simulator.pause()
    #simulator.plot()
    simulator.quit()





if __name__ == '__main__':
    #plot("test.csv")
    run()