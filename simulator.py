import os, csv
import importlib
from environment import TrafficLight, Environment, Car
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)

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
        ( 1,  0): 0, #East
        (-1,  0): 180, #West
        ( 0,  1): -90, #South
        ( 0, -1): 90 #North
    }

    def __init__(self, env, update_delay=0.5, display=True, filename="data.csv"):
        self.datafile = filename
        log_file = open(filename, 'wb')
        self.logger = csv.DictWriter(log_file, fieldnames=["trial", "cars", "total_stall", "average", "score"])
        self.logger.writeheader()
        self.trial = 0
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

        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay  # duration between each step (in seconds)
        self.loginfo = {}
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

    def run(self, trial, period):
        self.trial = trial
        gameExit = False
        clock = self.pygame.time.Clock()
        self.pygame.time.set_timer(self.pygame.USEREVENT , int(period*60000))
        while not gameExit:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.pygame.quit()
                elif event.type == self.pygame.USEREVENT:
                    gameExit = True

            self.screen.fill(self.bg_color)
            self.render()
            self.env.tick()
            self.pygame.display.flip()
            self.pygame.time.delay(self.frame_delay)
        if self.logger is not None:
            self.logger.writerow(self.loginfo)

    def quit(self):
        """ When the GUI is enabled, this function will pause the simulation. """
        self.font = self.pygame.font.Font(None, 30)
        paused = True
        pause_text = "Simulation Finished. Press any key to quit. . ."
        self.screen.blit(self.font.render(pause_text, True, self.colors['red'], self.bg_color), (400, self.height - 30))
        self.pygame.display.flip()
        print pause_text
        while paused:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.pygame.quit()
                if event.type == self.pygame.KEYDOWN:
                    paused = False
            self.pygame.time.wait(self.frame_delay)
        self.screen.blit(self.font.render(pause_text, True, self.bg_color, self.bg_color), (400, self.height - 30))
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
        ncars = 0
        for pos, obj in self.env.roads.iteritems():
            # Draw agent sprite (image), properly rotated
            direction, t = obj
            if type(t) is Car:
                ncars +=1
                _sprite = self.pygame.transform.smoothscale(
                    self.pygame.image.load(os.path.join("images", "car-{}.png".format(t.color))),
                    self.agent_sprite_size)
                px = self.block_size / 2 + ((pos[0] - 1) * 30)
                py = self.block_size + ((pos[1] - 1) * 30)
                rotated_sprite = self.pygame.transform.rotate(_sprite, self.rotation[t.get_heading()])
                self.screen.blit(rotated_sprite, self.pygame.rect.Rect(px ,py ,self.agent_sprite_size[0], self.agent_sprite_size[1]))

        # * Overlays
        self.font50 = self.pygame.font.Font(None, 50)
        self.screen.blit(
            self.font50.render("Training Trial %s (%s cars)" % (self.trial, ncars), True, self.colors['black'], self.bg_color),
            (10, 10))
        self.font = self.pygame.font.Font(None, 30)
        self.screen.blit(self.font.render("Number of Car Stall: %s" % self.env.stall, True, self.colors['dgreen'], self.bg_color),
                             (300, 50))
        average = round(self.env.totalStall/(self.env.t * 1.0), 2)
        score = round((ncars - average)/ncars * 100, 2)
        self.screen.blit(self.font.render("Total Stall: %s, Switch time: %s, Average : %s" % (self.env.totalStall, self.env.t, average) , True, self.colors['blue'], self.bg_color),
                             (300, 80))
        self.screen.blit(
            self.font50.render("Score: %s" % score, True, self.colors['magenta'], self.bg_color), (self.width - 300, self.height - 100))
        self.loginfo = {
            "trial" : self.trial,
            "cars" : ncars,
            "total_stall":self.env.totalStall,
            "average": average,
            "score" : score
        }

    def pause(self):
        """ When the GUI is enabled, this function will pause the simulation. """
        self.font = self.pygame.font.Font(None, 30)
        paused = True
        pause_text = "Simulation Paused. Press any key to continue. . ."
        self.screen.blit(self.font.render(pause_text, True, self.colors['red'], self.bg_color), (400, self.height - 30))
        self.pygame.display.flip()
        print pause_text
        while paused:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.pygame.quit()
                if event.type == self.pygame.KEYDOWN:
                    paused = False
            self.pygame.time.wait(self.frame_delay)
        self.screen.blit(self.font.render(pause_text, True, self.bg_color, self.bg_color), (400, self.height - 30))



