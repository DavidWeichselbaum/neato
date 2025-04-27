import math
import random

import pygame

from NEAT.utils import create_random_net
from agents.environment import Environment
from agents.constants import *
from agents.agent import Agent


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = Environment()

    for _ in range(10):
        start_position = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
        net = create_random_net(N_RANGEFINDERS, 2, 1, 4)
        energy = MAX_ENERGY * 0.5
        agent = Agent(start_position, net, energy)
        env.add_agent(agent)

    for _ in range(NUM_FOOD_INITIAL):
        env.spawn_food()

    food_timer = 0.0
    selected_agent = env.agents[0] if env.agents else None

    running = True
    while running:
        clock.tick(FPS)  # control FPS (but ignore real wall time)
        dt = 1.0 / FPS   # fixed simulation timestep
        food_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dead_agents, new_agents = env.update_agents()

        if selected_agent in dead_agents:
            selected_agent = env.agents[0] if env.agents else None

        for agent in env.agents:
            eaten = env.check_food_collisions(agent)
            for food in eaten:
                env.remove_food(food)

        if food_timer >= FOOD_SPAWN_INTERVAL:
            env.spawn_food()
            food_timer = 0.0

        env.step(dt)
        env.draw_environment(screen, selected_agent)
        pygame.display.flip()

        if not env.agents:
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()
