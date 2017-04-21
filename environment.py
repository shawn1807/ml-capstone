import random
from collections import OrderedDict


class TrafficLight(object):
    """A traffic light that switches periodically."""
    NS = "NS"
    EW = "EW"
    validate_ways = [NS, EW]  # NS=North South EW=East West

    def __init__(self, open_way=None):
        self.open_way = open_way if open_way is not None else random.choice(self.validate_ways)

    def switch(self):
        if self.open_way == self.NS:
            self.open_way = self.EW
        else:
            self.open_way = self.NS

    def get_open_way(self):
        return self.open_way


class TrafficLightControl(object):

    def __init__(self):
        pass

    def build(self):
        pass

    def update(self):
        pass


class Sensor(object):

    def __init__(self, env):
        self.env = env

    def sense(self):
        return {"light": 'red'}


class Car(object):
    color_choices = ['cyan', 'red', 'blue', 'green', 'orange', 'magenta', 'yellow']
    valid_actions = [None, 'forward', 'left', 'right']

    def __init__(self, env):
        self.env = env
        self.color = random.choice(self.color_choices)

    def update(self):
        pass


class RandomDrivingCar(Car):

    def __init__(self, env):
        super(RandomDrivingCar, self).__init__(env)
        self.sensor = Sensor(env)

    def update(self):
        info = self.sensor.sense()
        self.waypoint = random.choice(Car.valid_actions[1:])


class Environment(object):

    """Refer from 'train smartcab to drive' project."""

    def __init__(self, control=None, verbose=False, grid_size=(8, 6)):
        self.verbose = verbose
        self.grid_size = grid_size
        self.control = control

        # Road network
        self.grid_size = grid_size  # (columns, rows)
        self.bounds = (1, 2, self.grid_size[0], self.grid_size[1] + 1)
        self.block_size = 100
        self.hang = 0.6
        self.intersections = OrderedDict()
        self.roads = OrderedDict()
        for x in xrange(1, self.grid_size[0] + 1):
            for y in xrange(1, self.grid_size[1] + 1):
                self.intersections[(x, y)] = self.control.build() if self.control is not None else None # A traffic light at each intersection

        # Add environment roads
        for y in xrange(1, (self.grid_size[1]*3) + 2):
            for x in xrange(1, ((self.grid_size[0]*3) + 2)):
                rx = x % 3
                ry = y % 3
                if (rx == 1 or ry == 1) and rx != ry:
                    self.roads[(x, y)] = None



