import pymunk
from agents.constants import FOOD_COLOR, FOOD_SIGHT_COLOR, FOOD_RADIUS

class Food:
    def __init__(self, space, position):
        self.space = space
        self.color = FOOD_COLOR
        self.sight_color = FOOD_SIGHT_COLOR

        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, FOOD_RADIUS)
        self.shape.sensor = True
        self.shape.owner = self

        space.add(self.body, self.shape)
