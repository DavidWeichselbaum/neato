import math

import pygame
import random
import pymunk

from agents.agent import Agent
from agents.environment import Environment
from agents.constants import *


def make_dummy_net():
    class DummyNet:
        def activate(self, inputs):
            return [random.random() * 2 - 1, random.random() * 2 - 1]
    return DummyNet()


def draw_environment(screen, env, agents, selected_agent):
    screen.fill(BACKGROUND_COLOR)

    # Draw walls and food
    for shape in env.space.shapes:
        if isinstance(shape, pymunk.Circle):
            pos = int(shape.body.position.x), int(shape.body.position.y)
            radius = int(shape.radius)
            color = FOOD_COLOR if shape.sensor else AGENT_COLOR
            pygame.draw.circle(screen, color, pos, radius)
        elif isinstance(shape, pymunk.Segment):
            p1 = int(shape.a.x), int(shape.a.y)
            p2 = int(shape.b.x), int(shape.b.y)
            pygame.draw.line(screen, WALL_COLOR, p1, p2, WALL_THICKNESS)

    # Draw agents and facing direction
    for agent in agents:
        pos = int(agent.body.position.x), int(agent.body.position.y)
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
            hits = env.space.segment_query(start, end_point, 1, pymunk.ShapeFilter())
            min_end = end_point
            for hit in hits:
                if hit.shape != selected_agent.shape:
                    min_end = start + ray_dir * (hit.alpha * RANGEFINDER_RADIUS)
                    break
            pygame.draw.line(screen, RAY_COLOR, pos, (int(min_end.x), int(min_end.y)), 1)

        # Draw energy bar
        energy_ratio = selected_agent.energy / INITIAL_ENERGY
        pygame.draw.rect(screen, (255, 0, 0), (10, 10, 100, 10))
        pygame.draw.rect(screen, (0, 255, 0), (10, 10, int(100 * energy_ratio), 10))

        # Draw inputs and outputs
        font = pygame.font.SysFont(None, 24)
        y = 30
        for idx, val in enumerate(selected_agent.last_inputs):
            text = font.render(f'I{idx}: {val:.2f}', True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += 20
        for idx, val in enumerate(selected_agent.last_outputs):
            text = font.render(f'O{idx}: {val:.2f}', True, (255, 255, 0))
            screen.blit(text, (10, y))
            y += 20

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = Environment()

    agents = [Agent(env.space, (random.randint(100, WIDTH-100), random.randint(100, HEIGHT-100)), make_dummy_net()) for _ in range(10)]
    selected_agent = agents[0]

    # Spawn initial food
    for _ in range(NUM_FOOD):
        env.spawn_food()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for agent in agents:
            agent.update(env.space)

        for agent in agents:
            for food in env.food_shapes:
                if food.body.position.get_distance(agent.body.position) < AGENT_RADIUS + FOOD_RADIUS:
                    agent.energy += ENERGY_PER_FOOD
                    agent.energy = min(agent.energy, INITIAL_ENERGY)
                    # Relocate food
                    food.body.position = random.randint(20, WIDTH-20), random.randint(20, HEIGHT-20)

        env.space.step(1 / FPS)
        draw_environment(screen, env, agents, selected_agent)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
