import pymunk
import pygame

from agents.constants import WALL_COLOR, WALL_SIGHT_COLOR, WALL_THICKNESS


class Wall:
    def __init__(self, space, start, end):
        self.space = space
        self.color = WALL_COLOR
        self.sight_color = WALL_SIGHT_COLOR

        self.shape = pymunk.Segment(space.static_body, start, end, WALL_THICKNESS)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.8
        self.shape.owner = self

        space.add(self.shape)

    def draw(self, screen):
        p1 = int(self.shape.a.x), int(self.shape.a.y)
        p2 = int(self.shape.b.x), int(self.shape.b.y)
        pygame.draw.line(screen, self.color, p1, p2, WALL_THICKNESS)
