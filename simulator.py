import os
import time
import random
import importlib
from environment import TrafficLight, Environment, Car


class Simulator(object):
    """Simulates agents in a dynamic smartcab environment.

    Uses PyGame to display GUI, if available.
    """

    colors = {
        'black'   : (  0,   0,   0),
        'white'   : (255, 255, 255),
        'red'     : (255,   0,   0),
        'green'   : (  0, 255,   0),
        'dgreen'  : (  0, 228,   0),
        'blue'    : (  0,   0, 255),
        'cyan'    : (  0, 200, 200),
        'magenta' : (200,   0, 200),
        'yellow'  : (255, 255,   0),
        'mustard' : (200, 200,   0),
        'orange'  : (255, 128,   0),
        'maroon'  : (200,   0,   0),
        'crimson' : (128,   0,   0),
        'gray'    : (155, 155, 155)
    }

    rotation = {
        Environment.TE: 0,
        Environment.TW: 180,
        Environment.TN: 90,
        Environment.TS: -90
    }
    def __init__(self, env, update_delay=1.0, display=True):
        self.env = env
        self.block_size = 120
        self.size = ((env.grid_size[0] + 1) * self.block_size, (env.grid_size[1]+2) * self.block_size)
        self.width, self.height = self.size
        self.road_width = 60

        self.bg_color = self.colors['gray']
        self.road_color = self.colors['black']
        self.line_color = self.colors['mustard']
        self.boundary = self.colors['black']
        self.stop_color = self.colors['crimson']

        self.quit = False
        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay  # duration between each step (in seconds)

        self.display = display
        if self.display:
            try:
                self.pygame = importlib.import_module('pygame')
                self.pygame.init()
                self.screen = self.pygame.display.set_mode(self.size)

                self._ew = self.pygame.transform.smoothscale(
                    self.pygame.image.load(os.path.join("images", "east-west.png")), (self.road_width, self.road_width))
                self._ns = self.pygame.transform.smoothscale(
                    self.pygame.image.load(os.path.join("images", "north-south.png")),
                    (self.road_width, self.road_width))

                self.frame_delay = max(1, int(self.update_delay * 1000))  # delay between GUI frames in ms (min: 1)
                self.agent_sprite_size = (30, 30)

                self.agent_circle_radius = 20  # radius of circle, when using simple representation
                self.font = self.pygame.font.Font(None, 20)
                self.paused = False
            except ImportError as e:
                self.display = False
                print "Simulator.__init__(): Unable to import pygame; display disabled.\n{}: {}".format(
                    e.__class__.__name__, e)
            except Exception as e:
                self.display = False
                print "Simulator.__init__(): Error initializing GUI objects; display disabled.\n{}: {}".format(
                    e.__class__.__name__, e)

    def run(self):
        gameExit = False
        clock = self.pygame.time.Clock()
        while not gameExit:

            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    gameExit = True

            # Reset the screen.
            self.screen.fill(self.bg_color)

            # Draw elements
            # * Static elements

            # Boundary
            self.render()
            self.pygame.display.flip()
            self.pygame.time.delay(self.frame_delay)
        self.pygame.quit()

    def render(self):
        """ This is the GUI render display of the simulation. 
            Supplementary trial data can be found from render_text. """

        size = (self.block_size / 2, self.block_size, self.env.grid_size[0] * self.block_size ,self.env.grid_size[1] * self.block_size)
        self.pygame.draw.rect(self.screen, self.boundary, size, 4)
        # draw vertical road
        for i in xrange(1, self.env.grid_size[0]+1):
            startX = size[0] + (self.block_size * (i-0.5))  # self.block_size * i - self.block_size/2
            self.pygame.draw.line(self.screen, self.road_color, (startX, size[1]), (startX, size[1] + size[3]), self.road_width)
            self.pygame.draw.line(self.screen, self.line_color, (startX, size[1]), (startX, size[1] + size[3]), 2)
        # draw horizontal road
        for i in xrange(1, self.env.grid_size[1]+1):
            startY = size[1] + (self.block_size * (i-0.5))
            self.pygame.draw.line(self.screen, self.road_color, (size[0], startY), (size[0] + size[2], startY), self.road_width)
            self.pygame.draw.line(self.screen, self.line_color, (size[0], startY), (size[0] + size[2], startY), 2)

        for x in xrange(1, self.env.grid_size[0] + 1):
            startX = size[0] + (self.block_size * (x - 0.5)) - self.road_width / 2
            for y in xrange(1, self.env.grid_size[1] + 1):
                startY = size[1] + (self.block_size * (y - 0.5)) - self.road_width / 2
                #traffic light
                light = self.env.intersections[(x, y)]
                light.switch()
                square = (startX , startY, self.road_width,self.road_width)
                self.pygame.draw.rect(self.screen, self.boundary, square) # override background
                if light.get_open_way() == TrafficLight.NS:
                    self.screen.blit(self._ns, self.pygame.rect.Rect(square))
                    self.pygame.draw.line(self.screen, self.stop_color, (startX, startY), (startX , startY + self.road_width), 2)
                    self.pygame.draw.line(self.screen, self.stop_color, (startX + self.road_width, startY),(startX + self.road_width , startY + self.road_width), 2)
                else:
                    self.screen.blit(self._ew, self.pygame.rect.Rect(square))
                    self.pygame.draw.line(self.screen, self.stop_color, (startX, startY),
                                          (startX + self.road_width, startY), 2)
                    self.pygame.draw.line(self.screen, self.stop_color, (startX, startY + self.road_width),
                                          (startX + self.road_width, startY + self.road_width), 2)
        """draw cars"""
        for pos, obj in self.env.roads.iteritems():
            # Draw agent sprite (image), properly rotated
            direction, t = obj
            if type(t) is Car:
                _sprite = self.pygame.transform.smoothscale(
                    self.pygame.image.load(os.path.join("images", "car-{}.png".format('red'))),
                    self.agent_sprite_size)
                px = self.block_size / 2 + ((pos[0] - 1) * 30)
                py = self.block_size + ((pos[1] - 1) * 30 )
                rotated_sprite = self.pygame.transform.rotate(_sprite, self.rotation[direction])
                self.screen.blit(rotated_sprite, self.pygame.rect.Rect(px,py,self.agent_sprite_size[0], self.agent_sprite_size[1]))


