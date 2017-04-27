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

    def reset(self):
        self.last_updated = 0


def run():
    trials = 100
    cars = [50, 100, 150, 200, 250, 300]
    period = 0.5
    agent = BenchmarkAgent()
    env = Environment(control=agent, grid_size=(8, 6))
    simulator = Simulator(env, update_delay=0.1, filename="benchmark.csv")
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
import pandas
import matplotlib.pyplot as plt

def plot(datafile):
    plt.figure()
    df = pandas.read_csv(datafile)
    df = df[["score","average","cars", "total_stall"]].groupby("cars")
    print df.describe()
    #df.plot.box()
    #plt.show()




if __name__ == '__main__':
    #plot("test.csv")
    run()