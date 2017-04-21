from environment import Environment, TrafficLightControl, TrafficLight
from simulator import Simulator

class BenchmarkAgent(TrafficLightControl):
    """ represent benchmark agent"""

    def __init__(self, period=2):
        self.period = period
        self.lights = []
        self.last_updated = 0

    def build(self):
        tl = TrafficLight()
        self.lights.append(tl)
        return tl

    def update(self, t):
        if t - self.last_updated >= self.period:
            for l in self.lights:
                l.switch()
            self.last_updated = t


def run():
    agent = BenchmarkAgent()
    env = Environment(control=agent)
    simulator = Simulator(env)
    simulator.run()

if __name__ == '__main__':
    run()
