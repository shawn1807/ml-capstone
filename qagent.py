from environment import Environment, TrafficLightControl, TrafficLight, Car
from simulator import Simulator
import random, numpy as np
from collections import OrderedDict
from ann import NeuralNetwork, NeuronInitializer

local_reward = 1
"""
one single traffic light pairs two neurons used to judge q value
"""
class NeuronTrafficLight(TrafficLight):

    def __init__(self, open_way=None):
        super(NeuronTrafficLight, self).__init__(open_way)
        self.ns_neuron = None
        self.ew_neuron = None
        self.ns_rewards = 0
        self.ew_rewards = 0

    def switch(self):
        diff = self.ns_neuron.output - self.ew_neuron.output
        #Q value must at least great than 0.25 otherwise random choice one
        if diff > 0.25:
            self.open_way = self.NS
        elif diff < -0.25:
            self.open_way = self.NS
        else:
            self.open_way = random.choice([self.EW, self.NS])


class WeightsInitializer(NeuronInitializer):

    def init_weights(self, size):
        return np.random.randn(size)


class QLearningAgent(TrafficLightControl):
    """ Q learning network agent"""

    def __init__(self, period=2, learning=False, epsilon=0.7, alpha=0.5, batch_size=10, gamma=1):
        super(QLearningAgent, self).__init__()
        self.period = period
        self.lights = []
        self.lightPositions = OrderedDict()
        self.learning = learning
        self.epsilon = min([0.99, epsilon])
        self.alpha = alpha
        self.model = None
        self.input_x = None
        self.learning_count = 0
        self.batch_size = batch_size
        self.gamma = gamma
        # the state will stuck
        self.flag = False
        # (s, a, r, s')
        self.replay = []
        self.replay_size = 500
        self.actions = None

    def build_light(self):
        tl = NeuronTrafficLight()
        self.lights.append(tl)
        return tl

    def setup(self):
        self.model = NeuralNetwork(alpha=self.alpha)
        self.model.set_input_layer(len(self.env.roads))
        self.model.add_hidden_layer(len(self.lights)*4, activation="sigmoid")
        self.model.set_output_layer(len(self.lights)*2, activation="linear")
        self.model.build()
        for i in range(0, len(self.lights)):
            self.lights[i].ns_neuron = self.model.output_layer.neurons[i]
            self.lights[i].ew_neuron = self.model.output_layer.neurons[i * 2 + 1]

    def signal(self):
        if self.env.t % self.period == 0:
            self.input_x = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            self.model.signal(self.input_x)

    def after_signal(self):
        if self.env.t % self.period == 0 and self.learning:
            g_reward = self.env.number_of_car - self.env.stall
            if self.learning_count > 1000:
                self.learning = False
            next_state = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            rewards = []
            for l in self.lights:
                rewards.append(round(
                    l.ns_rewards + g_reward if l.ns_neuron.output < l.ns_rewards else 0 * 1. / self.env.number_of_car,
                    2))
                rewards.append(round(
                    l.ew_rewards + g_reward if l.ew_neuron.output < l.ew_rewards else 0 * 1. / self.env.number_of_car,
                    2))
                # reset local reward
                l.ns_rewards = 0
                l.ew_rewards = 0
            self.model.back_propagate(rewards)
            self.model.update()

    def signal2(self):
        if self.env.t % self.period == 0:
            self.input_x = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            self.model.signal(self.input_x)
            if not self.learning:
                self.actions = []
                for l in self.lights:
                    l.switch()
                    self.actions.append(l.open_way)
            else:
                self.actions = [random.choice([TrafficLight.NS, TrafficLight.EW]) for i in range(0, len(self.lights))]
                for a, l in zip(self.actions, self.lights):
                    l.open_way = a


    def after_signal2(self):
        if self.env.t % self.period == 0 and self.learning:
            self.learning_count += 1
            #global reward
            g_reward = self.env.number_of_car - self.env.stall
            if self.learning_count > 1000:
                self.learning = False
            next_state = np.array([0 if o is None else 1 for p, o in self.env.roads.values()])
            rewards = []
            for l in self.lights:
                rewards.append(round(l.ns_rewards + g_reward if l.ns_neuron.output < l.ns_rewards else 0 * 1./self.env.number_of_car, 2))
                rewards.append(round(l.ew_rewards + g_reward if l.ew_neuron.output < l.ew_rewards else 0 * 1./self.env.number_of_car, 2))
                # reset local reward
                l.ns_rewards = 0
                l.ew_rewards = 0
            if len(self.replay) < self.replay_size:
                self.replay.append((self.input_x, self.actions, rewards, next_state))
            else:
                #mini batch
                batch = random.sample(self.replay, self.batch_size)
                for b in batch:
                    s0, a0, r0, s1 = b
                    # find next s1 state return first, because s0 state needs propagate error back and neural network kept last activation
                    q1 = self.model.signal(s1)
                    q0 = self.model.signal(s0)
                    expected = []
                    for i in range(0, len(q0), 2):
                        q0_ns = q0[i]
                        q0_ew = q0[i+1]
                        maxQ = np.max(q1[i:i+2])
                        #if reward is 0 means not taking this action
                        if r0[i] == 0:
                            expected.append(0.)
                            r = np.min([1, r0[i+1] + (self.gamma * maxQ)])
                            expected.append(r)
                        else:
                            r = np.min([1, r0[i] + (self.gamma * maxQ)])
                            expected.append(r)
                            expected.append(0.)
                    self.model.back_propagate(expected)
                    print "q0", q0
                    print "expected", expected
                    print self.model.signal(s0)
                #batch update
                self.model.update()
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon - (1./((self.learning_count+1) ** 2))

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
                    light.ns_rewards += local_reward
                else:
                    light.ew_rewards += local_reward
            else:
                allowPassing = heading == (1, 0) or heading == (-1, 0)
                if allowPassing:
                    light.ew_rewards += local_reward
                else:
                    light.ns_rewards += local_reward
        return super(QLearningAgent,self).allow(position,heading)


def run():
    trials = 2
    cars = [50,100]
    period = 50
    agent = QLearningAgent(learning=True, alpha=0.7, epsilon=1,batch_size=1)
    env = Environment(control=agent, grid_size=(8,4))
    simulator = Simulator(env, update_delay=0.1, filename="qagent.csv")
    simulator.title = "Training Learning Q-Agent"
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
        agent.learning = False
    simulator.quit()

if __name__ == '__main__':
    run()
