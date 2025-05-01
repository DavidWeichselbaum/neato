import math
import random
from time import time

import pygame
import matplotlib.pyplot as plt
from datetime import timedelta

from NEAT.utils import create_random_net
from agents.environment import Environment
from agents.constants import *
from agents.agent import Agent


def setup_env():
    env = Environment()

    for _ in range(NUM_AGENTS_INITIAL):
        start_position = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
        net = create_random_net(
            n_inputs=N_INPUTS, n_outputs=2, n_hidden=1, n_connections=10,
            output_activation_choices=["tanh"],  # range: (-1, 1)
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        )
        energy = MAX_ENERGY * 0.5
        facing_angle = random.uniform(0, 2 * math.pi)
        agent = Agent(start_position, net, energy, facing_angle)
        env.add_agent(agent)

    for _ in range(NUM_FOOD_INITIAL):
        env.spawn_food()

    return env


def handle_selected_agent(selected_agent, dead_agents, keys, env, fig, ax):
    if not selected_agent or selected_agent in dead_agents:
        selected_agent = env.agents[0] if env.agents else None

        ax.clear()
        if selected_agent:
            selected_agent.net.get_graph_plot(ax)
        fig.canvas.draw()
        fig.canvas.flush_events()

    controll_agent(keys, selected_agent)
    return selected_agent


def controll_agent(keys, selected_agent):
    possession_key = keys[pygame.K_SPACE]
    forward_key = keys[pygame.K_w]
    backward_key = keys[pygame.K_s]
    left_key = keys[pygame.K_a]
    right_key = keys[pygame.K_d]

    if possession_key:
        selected_agent.possessed = not selected_agent.possessed

    acceleration = 0.0
    rotation = 0.0
    if forward_key:
        acceleration = 1.0
    if backward_key:
        acceleration = -1.0
    if left_key:
        rotation = -1.0
    if right_key:
        rotation = 1.0
    selected_agent.possession_outputs = [acceleration, rotation]


def print_infos(env, clock, simulation_seconds, start_time):
    n_agents = len(env.agents)
    n_foods = len(env.foods)
    visualization_FPS = clock.get_fps()
    simulation_time = timedelta(seconds=int(simulation_seconds))
    wall_time = timedelta(seconds=int(time() - start_time))
    print(f"Sim time: {simulation_time}    Wall time: {wall_time}    FPS: {visualization_FPS:.1f}    Agents: {n_agents}    Food: {n_foods}")


def simulation_step(dt, env, food_timer):
    dead_agents, new_agents = env.update_agents(dt)
    env.update_food()
    env.step(dt)

    food_timer += dt
    if food_timer >= FOOD_SPAWN_INTERVAL:
        env.spawn_food()
        food_timer = 0.0

    return food_timer, dead_agents

def visualization_step(screen, running, clock, selected_agent, dead_agents, env, fig, ax, disable_rendering):
    keys = pygame.key.get_pressed()

    in_background = keys[pygame.K_b]
    in_foreground = keys[pygame.K_f]
    if in_background:
        disable_rendering = True
    if in_foreground:
        disable_rendering = False
        selected_agent = None

    if not disable_rendering:
        clock.tick(FPS_VISUALIZATION)
        selected_agent = handle_selected_agent(selected_agent, dead_agents, keys, env, fig, ax)

        screen.fill(BACKGROUND_COLOR)
        env.draw_environment(screen)
        if selected_agent:
            selected_agent.draw_annotation(screen, env)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    return running, selected_agent, disable_rendering




def main():
    env = setup_env()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))

    food_timer, info_timer = 0.0, 0.0
    selected_agent = None
    disable_rendering = False
    running = True
    start_time = time()
    simulation_seconds = 0.0
    dt = 1.0 / FPS_SIMULATION   # fixed simulation timestep
    while running:
        simulation_seconds += dt

        food_timer, dead_agents = simulation_step(dt, env, food_timer)
        if not env.agents:
            running = False

        info_timer += dt
        if info_timer >= INFO_INTERVAL:
            print_infos(env, clock, simulation_seconds, start_time)
            info_timer = 0.0

        running, selected_agent, disable_rendering = visualization_step(
            screen, running, clock, selected_agent, dead_agents, env, fig, ax, disable_rendering)

    pygame.quit()
    plt.close(fig)

if __name__ == "__main__":
    pygame.init()
    main()
