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

    def spawn_food(self):
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

    def update_agents(self):
        new_agents = []
        dead_agents = []

        for agent in self.agents:

            if agent.energy <= 0:
                dead_agents.append(agent)
                continue

            if agent.energy >= MAX_ENERGY:
                agent.energy = MAX_ENERGY * 0.5
                facing = pymunk.Vec2d(math.cos(agent.facing_angle), math.sin(agent.facing_angle))
                spawn_position = agent.body.position - facing * (AGENT_RADIUS * 4)

                while True:
                    try:
                        child_net = mutate_net(agent.net)
                        break
                    except ValueError as error:
                        print(f"Mutation failed due to: {error}")

                energy = MAX_ENERGY * 0.5
                child_agent = Agent(spawn_position, child_net, energy)
                child_agent.facing_angle = (agent.facing_angle + math.pi) % (2 * math.pi)
                new_agents.append(child_agent)

            agent.update(self.space)

        for agent in dead_agents:
            self.remove_agent(agent)

        for agent in new_agents:
            self.add_agent(agent)

        return dead_agents, new_agents

    def check_food_collisions(self, agent):
        eaten = []
        for food in self.foods:
            if food.body.position.get_distance(agent.body.position) < AGENT_RADIUS + FOOD_RADIUS:
                agent.energy += ENERGY_PER_FOOD
                agent.energy = min(agent.energy, MAX_ENERGY)
                eaten.append(food)
        return eaten

    def step(self, dt):
        self.space.step(dt)

    def draw_environment(self, screen, selected_agent):
        screen.fill(BACKGROUND_COLOR)

        # Draw walls
        for wall in self.walls:
            p1 = int(wall.shape.a.x), int(wall.shape.a.y)
            p2 = int(wall.shape.b.x), int(wall.shape.b.y)
            pygame.draw.line(screen, wall.color, p1, p2, WALL_THICKNESS)

        # Draw food
        for food in self.foods:
            pos = int(food.body.position.x), int(food.body.position.y)
            pygame.draw.circle(screen, food.color, pos, int(food.shape.radius))

        # Draw agents
        for agent in self.agents:
            pos = int(agent.body.position.x), int(agent.body.position.y)
            pygame.draw.circle(screen, agent.color, pos, AGENT_RADIUS)
            # Draw facing direction
            facing = pymunk.Vec2d(math.cos(agent.facing_angle), math.sin(agent.facing_angle))
            end = pos[0] + int(facing.x * 20), pos[1] + int(facing.y * 20)
            pygame.draw.line(screen, FACING_COLOR, pos, end, 2)

        # Draw selected agent's sensors and HUD
        if selected_agent:
            pos = int(selected_agent.body.position.x), int(selected_agent.body.position.y)
            angle_start = selected_agent.facing_angle - math.pi / 2
            angle_step = math.pi / (N_RANGEFINDERS - 1)
            for i in range(N_RANGEFINDERS):
                angle = angle_start + i * angle_step
                ray_dir = pymunk.Vec2d(math.cos(angle), math.sin(angle))
                start = selected_agent.body.position
                end_point = start + ray_dir * RANGEFINDER_RADIUS
                hits = self.space.segment_query(start, end_point, 1, pymunk.ShapeFilter())
                min_end = end_point
                for hit in hits:
                    if hit.shape != selected_agent.shape:
                        min_end = start + ray_dir * (hit.alpha * RANGEFINDER_RADIUS)
                        break
                pygame.draw.line(screen, RAY_COLOR, pos, (int(min_end.x), int(min_end.y)), 1)

            # Draw energy bar
            energy_ratio = selected_agent.energy / MAX_ENERGY
            pygame.draw.rect(screen, (255, 0, 0), (10, 10, 100, 10))
            pygame.draw.rect(screen, (0, 255, 0), (10, 10, int(100 * energy_ratio), 10))

            # Draw inputs and outputs
            font = pygame.font.SysFont(None, 24)
            y = 30
            for idx, val in enumerate(selected_agent.last_inputs):
                input_idx = selected_agent.net.input_ids[idx]
                input_node = selected_agent.net.node_map[input_idx]
                input_name = input_node.name

                text = font.render(f'I{idx} | {input_name}: {val:.2f}', True, (0, 255, 255))
                screen.blit(text, (10, y))
                y += 20
            for idx, val in enumerate(selected_agent.last_outputs):
                output_idx = selected_agent.net.output_ids[idx]
                output_node = selected_agent.net.node_map[output_idx]
                output_name = output_node.name

                text = font.render(f'O{idx} | {output_name}: {val:.2f}', True, (255, 0, 255))
                screen.blit(text, (10, y))
                y += 20
