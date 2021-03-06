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
        self.lightPositions = OrderedDict()
        self.env = None

    def build_light(self):
        pass

    def setup(self):
        pass

    def set_env(self, env):
        self.env = env

    def signal(self):
        pass

    def after_signal(self):
        pass

    def reset(self):
        pass

    def register(self, light, positions):
        for position in positions:
            self.lightPositions[position] = light

    def allow(self, position, heading):
        """already in intersections"""
        if self.lightPositions.has_key(position):
            return True
        """check next position if it is in the intersections"""
        pos = (position[0]+heading[0], position[1]+heading[1])
        if self.lightPositions.has_key(pos):
            light = self.lightPositions[pos]
            if light.get_open_way() == TrafficLight.NS:
                return heading == (0, 1) or heading == (0, -1)
            else:
                return heading == (1, 0) or heading == (-1, 0)
        else:
            return True


class Navigator(object):

    def __init__(self, car, env):
        self.env = env
        self.car = car
        self.direction = random.choice(env.roads[car.position][0])
        self.heading = self.direction

    def navigate(self):
        direction, obj = self.env.roads[self.car.position]
        self.direction = random.choice(direction)
        if not self.env.control.allow(self.car.position, self.direction):
            self.heading = (0, 0)
        else:
            self.heading = self.direction


class Car(object):

    color_choices = ['white', 'cyan', 'red', 'blue', 'green', 'orange', 'magenta', 'yellow']

    def __init__(self, env, position):
        self.env = env
        self.position = position
        self.color = random.choice(self.color_choices)
        self.navigator = Navigator(self, env)
        self.navigator.heading = random.choice(env.roads[position][0])
        env.roads[position] = (env.roads[position][0], self)
        self.state = 1 # 0: stall 1: moving
        self.mov = (0, 0)

    def step(self):
        self.state = 1
        self.navigator.navigate()
        if self.navigator.heading == (0, 0):
            self.state = 0
            return
        self.env.act(self, self.navigator.heading)

    def get_direction(self):
        return self.navigator.direction


class Environment(object):

    """Refer from 'train smartcab to drive' project."""
    TNW = ((0,  -1), (-1,  0)) # Toward North or West, north is two times important
    TNE = ((0,  -1), (1,  0)) # Toward North or East, north is two times important
    TSW = ((0,  1), (-1,  0)) # Toward South or West
    TSE = ((0,  1), (1,  0)) # Toward South or East
    TN = ((0,  -1),) # Toward North
    TS = ((0, 1),) # Toward South
    TE = ((1,  0),) # Toward East
    TW = ((-1, 0),) # Toward West

    def __init__(self, control=None, verbose=False, grid_size=(8, 6)):
        self.verbose = verbose
        self.grid_size = grid_size
        self.control = control
        self.control.set_env(self)
        self.bounds = (1, 1, self.grid_size[0] * 4 , self.grid_size[1] * 4)
        self.grid_size = grid_size  # (columns, rows)
        self.number_of_car = 0
        # Road network
        self.intersections = OrderedDict()
        self.roads = OrderedDict()
        self.t = 1
        self.stall = 0
        self.totalStall = 0
        """
        road template :
        X |TS    |TN    |X
        TW|L(TSW)|L(TNW)|TW
        TE|L(TSE)|L(TNE)|TE
        X |TS    |TN    |X        
        """
        for i in xrange(1, self.grid_size[1] + 1):
            for j in xrange(1, self.grid_size[0] + 1):
                light = self.control.build_light() if self.control is not None else None
                self.intersections[j, i] = light
                x = (j - 1) * 4
                y = (i - 1) * 4
                self.roads[(x + 2, y + 1)] = (self.TS, None)
                self.roads[(x + 3, y + 1)] = (self.TN, None)
                self.roads[(x + 1, y + 2)] = (self.TW, None)
                self.roads[(x + 2, y + 2)] = (self.TSW, None)
                self.roads[(x + 3, y + 2)] = (self.TNW, None)
                self.roads[(x + 4, y + 2)] = (self.TW, None)
                self.roads[(x + 1, y + 3)] = (self.TE, None)
                self.roads[(x + 2, y + 3)] = (self.TSE, None)
                self.roads[(x + 3, y + 3)] = (self.TNE, None)
                self.roads[(x + 4, y + 3)] = (self.TE, None)
                self.roads[(x + 2, y + 4)] = (self.TS, None)
                self.roads[(x + 3, y + 4)] = (self.TN, None)
                self.control.register(light, [(x + 2, y + 2),(x + 3, y + 2), (x + 2, y + 3), (x + 3, y + 3)])
        self.control.setup()

    def reset(self):
        self.t = 1
        self.stall = 0
        self.totalStall = 0
        self.control.reset()
        for pos in self.roads.keys():
            d, obj = self.roads[pos]
            self.roads[pos] = (d, None)

    def tick(self):
        self.t += 1
        self.stall = 0
        self.control.signal()
        self.number_of_car = 0
        for pos, t in self.roads.items():
            pos, obj = t
            if type(obj) is Car:
                self.number_of_car += 1
                obj.step()
                if obj.state == 0:
                    self.stall += 1
        self.totalStall += self.stall
        self.control.after_signal()

    def act(self, car, mov):
        way_point = self.roads[car.position][0]
        next_pos = (car.position[0] + mov[0], car.position[1] + mov[1])
        """if run into boundary then reset position to other side"""
        if self.bounds[0] > next_pos[0] or next_pos[0] > self.bounds[2]:
            next_pos = (abs(next_pos[0] - self.bounds[2]), next_pos[1])
        elif self.bounds[1] > next_pos[1] or next_pos[1] > self.bounds[3]:
            next_pos = (next_pos[0], abs(next_pos[1] - self.bounds[3]))
        direction, obj = self.roads[next_pos]
        """ only update position when there is no car ahead"""
        if obj is None:
            self.roads[car.position] = (way_point, None)
            self.roads[next_pos] = (direction, car)
            car.position = next_pos
        else:
            car.state = 0

