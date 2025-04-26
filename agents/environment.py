import pymunk
import random
import math

from agents.constants import *


class Environment:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.food_shapes = []

        self.create_boundary_walls()
        for _ in range(NUM_WALLS):
            self.create_random_wall()

    def create_random_wall(self):
        x1 = random.randint(0, WIDTH)
        y1 = random.randint(0, HEIGHT)
        length = random.randint(WALL_MIN_LEN, WALL_MAX_LEN)
        angle = random.uniform(0, 2 * math.pi)
        x2 = x1 + math.cos(angle) * length
        y2 = y1 + math.sin(angle) * length
        segment = pymunk.Segment(self.space.static_body, (x1, y1), (x2, y2), WALL_THICKNESS)
        segment.elasticity = WALL_ELASTICITY
        segment.friction = WALL_FRICTION
        self.space.add(segment)

    def create_boundary_walls(self):
        corners = [
            ((0, 0), (WIDTH, 0)),
            ((WIDTH, 0), (WIDTH, HEIGHT)),
            ((WIDTH, HEIGHT), (0, HEIGHT)),
            ((0, HEIGHT), (0, 0)),
        ]
        for a, b in corners:
            segment = pymunk.Segment(self.space.static_body, a, b, WALL_THICKNESS)
            segment.elasticity = WALL_ELASTICITY
            segment.friction = WALL_FRICTION
            self.space.add(segment)

    def spawn_food(self):
        for _ in range(100):
            pos = random.randint(20, WIDTH - 20), random.randint(20, HEIGHT - 20)
            if all(math.hypot(pos[0] - food.body.position.x, pos[1] - food.body.position.y) > FOOD_MIN_DIST for food in self.food_shapes):
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = pos
                shape = pymunk.Circle(body, FOOD_RADIUS)
                shape.sensor = True
                self.space.add(body, shape)
                self.food_shapes.append(shape)
                return
