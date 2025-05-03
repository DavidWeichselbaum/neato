import math
import random
from dataclasses import dataclass
from time import time

import pygame
import matplotlib.pyplot as plt
from datetime import timedelta

from NEAT.utils import create_random_net
from agents.environment import Environment
from agents.constants import *
from agents.agent import Agent


@dataclass
class Info():
    dt: float
    running: bool
    food_timer: float

    disable_rendering: bool
    info_timer: float
    selected_agent: Agent | None
    start_time: float
    simulation_seconds: float
    clock: pygame.time.Clock

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
        env.spawn_food(age=FOOD_INITIAL_AGE)

    return env



def init_info_object():
    dt = 1.0 / FPS_SIMULATION  # fixed simulation time step
    clock = pygame.time.Clock()
    start_time = time()
    return Info(
        dt=dt,
        running=True,
        food_timer=0.0,
        disable_rendering=False,
        info_timer=0.0,
        selected_agent=None,
        start_time=start_time,
        simulation_seconds=0.0,
        clock=clock,
    )

def handle_selected_agent(env, infos, keys, fig, ax):
    if not infos.selected_agent:
        infos.selected_agent = env.agents[0] if env.agents else None

        ax.clear()
        if infos.selected_agent:
            infos.selected_agent.net.get_graph_plot(ax)
        fig.canvas.draw()
        fig.canvas.flush_events()

    if infos.selected_agent:
        controll_agent(keys, infos.selected_agent)


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


def print_infos(env, infos):
    infos.info_timer += infos.dt
    if infos.info_timer >= INFO_INTERVAL:
        infos.info_timer = 0.0

        n_agents = len(env.agents)
        n_foods = len(env.foods)
        visualization_FPS = infos.clock.get_fps()
        simulation_time = timedelta(seconds=int(infos.simulation_seconds))
        wall_time = timedelta(seconds=int(time() - infos.start_time))
        print(f"Sim time: {simulation_time}    Wall time: {wall_time}    FPS: {visualization_FPS:.1f}    Agents: {n_agents}    Food: {n_foods}")



def simulation_step(env, infos):
    dead_agents, new_agents = env.update_agents(infos.dt)
    if not env.agents:
        infos.running = False

    env.update_food(infos.dt)
    infos.food_timer += infos.dt
    if infos.food_timer >= FOOD_SPAWN_INTERVAL:
        env.spawn_food()
        infos.food_timer = 0.0

    if infos.selected_agent in dead_agents:
        infos.selected_agent = None

    env.step(infos.dt)
    infos.simulation_seconds += infos.dt


def visualization_step(screen, env, infos, fig, ax):
    keys = pygame.key.get_pressed()

    in_background = keys[pygame.K_b]
    in_foreground = keys[pygame.K_f]
    if in_background:
        infos.disable_rendering = True
    if in_foreground:
        infos.disable_rendering = False
        infos.selected_agent = None

    if not infos.disable_rendering:
        infos.clock.tick(FPS_VISUALIZATION)
        handle_selected_agent(env, infos, keys, fig, ax)

        screen.fill(BACKGROUND_COLOR)
        env.draw_environment(screen)
        if infos.selected_agent:
            infos.selected_agent.draw_annotation(screen, env)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt


def main(screen, fig, ax):
    env = setup_env()

    infos = init_info_object()

    while infos.running:
        simulation_step(env, infos)

        print_infos(env, infos)

        visualization_step(screen, env, infos, fig, ax)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))

    while True:
        main(screen, fig, ax)
        print("Restarting simulation")

    pygame.quit()
    plt.close(fig)
