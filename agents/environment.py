import pymunk
import random
import math

import pygame

from NEAT.utils import mutate_net
from agents.agent import Agent
from agents.food import Food
from agents.wall import Wall
from agents.constants import *


class Environment:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.foods = []
        self.walls = []
        self.agents = []

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
        wall = Wall(self.space, (x1, y1), (x2, y2))
        self.walls.append(wall)

    def create_boundary_walls(self):
        corners = [
            ((0, 0), (WIDTH, 0)),
            ((WIDTH, 0), (WIDTH, HEIGHT)),
            ((WIDTH, HEIGHT), (0, HEIGHT)),
            ((0, HEIGHT), (0, 0)),
        ]
        for a, b in corners:
            wall = Wall(self.space, a, b)
            self.walls.append(wall)

    def spawn_food(self, age=0.0):
        pos = random.randint(20, WIDTH - 20), random.randint(20, HEIGHT - 20)
        food = Food(self.space, pos)
        self.foods.append(food)

    def remove_food(self, food):
        self.space.remove(food.body, food.shape)
        self.foods.remove(food)

    def add_agent(self, agent):
        self.agents.append(agent)
        self.space.add(agent.body, agent.shape)

    def remove_agent(self, agent):
        self.space.remove(agent.body, agent.shape)
        self.agents.remove(agent)

    def update_agents(self, dt):
        new_agents = []
        dead_agents = []

        for agent in self.agents:

            if agent.energy <= 0:
                dead_agents.append(agent)
                continue

            agent.update(self.space, dt)

            if agent.energy >= agent.reproduction_energy_threshold:
                child_agent = agent.reproduce()
                new_agents.append(child_agent)

        for agent in dead_agents:
            self.remove_agent(agent)

        for agent in new_agents:
            self.add_agent(agent)

        return dead_agents, new_agents

    def update_food(self, dt):
        for agent in self.agents:
            eaten = self.check_food_collisions(agent)
            for food in eaten:
                self.remove_food(food)

        for food in self.foods:
            food.age += dt
            if food.age > FOOD_MAX_AGE:
                self.remove_food(food)

    def check_food_collisions(self, agent):
        eaten = []
        for food in self.foods:
            if food.body.position.get_distance(agent.body.position) < AGENT_RADIUS + FOOD_RADIUS:
                agent.energy += FOOD_ENERGY
                agent.energy = min(agent.energy, MAX_ENERGY)
                eaten.append(food)
        return eaten

    def step(self, dt):
        self.space.step(dt)

    def draw_environment(self, screen):
        for wall in self.walls:
            wall.draw(screen)
        for food in self.foods:
            food.draw(screen)
        for agent in self.agents:
            agent.draw(screen)
