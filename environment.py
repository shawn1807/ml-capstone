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

    def __init__(self, env, position):
        self.env = env
        self.position = position
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
    TN = (0,  1) # Toward North
    TS = (0, -1) # Toward South
    TE = (1,  0) # Toward East
    TW = (-1, 0) # Toward West


    def __init__(self, control=None, verbose=False, grid_size=(8, 6)):
        self.verbose = verbose
        self.grid_size = grid_size
        self.control = control
        self.bounds = (1, 1, self.grid_size[0] * 4, self.grid_size[1] *4)
        # Road network
        self.grid_size = grid_size  # (columns, rows)
        self.intersections = OrderedDict()
        self.roads = OrderedDict()
        """
        road template 
        X |TS    |TN   |X
        TW|L(TS) |L(TW)|TW
        TE|L(TE) |L(TN)|TE
        X |TS    |TN   |X        
        """
        for i in xrange(1, self.grid_size[1] + 1):
            for j in xrange(1, self.grid_size[0] + 1):
                light = self.control.build() if self.control is not None else None
                self.intersections[(j, i)] = light
                x = (j - 1) * 4
                y = (i - 1) * 4
                self.roads[(x + 2, y + 1)] = (self.TS, None)
                self.roads[(x + 3, y + 1)] = (self.TN, None)
                self.roads[(x + 1, y + 2)] = (self.TW, None)
                self.roads[(x + 2, y + 2)] = (self.TS, light)
                self.roads[(x + 3, y + 2)] = (self.TW, light)
                self.roads[(x + 4, y + 2)] = (self.TW, None)
                self.roads[(x + 1, y + 3)] = (self.TE, None)
                self.roads[(x + 2, y + 3)] = (self.TE, light)
                self.roads[(x + 3, y + 3)] = (self.TN, light)
                self.roads[(x + 4, y + 3)] = (self.TE, None)
                self.roads[(x + 2, y + 4)] = (self.TS, None)
                self.roads[(x + 3, y + 4)] = (self.TN, None)
