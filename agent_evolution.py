import math
import random

import pygame
import matplotlib.pyplot as plt

from NEAT.utils import create_random_net
from agents.environment import Environment
from agents.constants import *
from agents.agent import Agent


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))

    env = Environment()

    for _ in range(10):
        start_position = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
        net = create_random_net(
            n_inputs=N_INPUTS, n_outputs=2, n_hidden=1, n_connections=10,
            output_activation_choices=["tanh"],  # range: (-1, 1)
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        )
        energy = MAX_ENERGY * 0.5
        agent = Agent(start_position, net, energy)
        env.add_agent(agent)

    for _ in range(NUM_FOOD_INITIAL):
        env.spawn_food()

    selected_agent = env.agents[0] if env.agents else None
    last_selected_net = None

    food_timer, info_timer = 0.0, 0.0
    running = True
    while running:
        clock.tick(FPS)
        dt = 1.0 / FPS   # fixed simulation timestep

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

        food_timer += dt
        if food_timer >= FOOD_SPAWN_INTERVAL:
            env.spawn_food()
            food_timer = 0.0

        info_timer += dt
        if info_timer >= INFO_INTERVAL:
            print(f"Agents: {len(env.agents)}, FPS: {clock.get_fps():.1f}")
            info_timer = 0.0

        env.step(dt)

        if not env.agents:
            running = False

        if selected_agent and (selected_agent.net != last_selected_net):
            ax.clear()
            selected_agent.net.get_graph_plot(ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            last_selected_net = selected_agent.net

        env.draw_environment(screen, selected_agent)
        pygame.display.flip()

    pygame.quit()
    plt.close(fig)

if __name__ == "__main__":
    pygame.init()
    main()
