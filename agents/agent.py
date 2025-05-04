import math
import random

import pymunk
import pygame
import numpy as np

from agents.constants import *
from NEAT.utils import mutate_net


class Agent:
    def __init__(self, position, net, energy, angle):
        self.net = net
        self.energy = energy

        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, AGENT_RADIUS))
        self.body.position = position
        self.body.angle = angle

        self.shape = pymunk.Circle(self.body, AGENT_RADIUS)
        self.shape.elasticity = WALL_ELASTICITY
        self.shape.friction = WALL_FRICTION
        self.shape.owner = self

        self.color = AGENT_COLOR
        self.sight_color = AGENT_SIGHT_COLOR

        self.counter = 0
        self.last_inputs = []
        self.last_outputs = []
        self.reproduction_energy_threshold = MAX_ENERGY * 0.5
        self.reproduction_energy_ratio = 0.5

        self.possessed = False
        self.possession_outputs = [0.0, 0.0]

    def draw(self, screen):
        pos = int(self.body.position.x), int(self.body.position.y)
        pygame.draw.circle(screen, self.color, pos, AGENT_RADIUS)

        # Draw facing direction
        facing = pymunk.Vec2d(math.cos(self.body.angle), math.sin(self.body.angle))
        end = pos[0] + int(facing.x * 20), pos[1] + int(facing.y * 20)
        pygame.draw.line(screen, FACING_COLOR, pos, end, 2)

    def update(self, space, dt):
        self.counter += 1
        thrust = 0.0
        if self.counter % NETWORK_EVALUATION_STEP == 0:
            rangefinder_inputs = self.get_rangefinder_inputs(space)
            random_inputs = self.get_random_input()
            inputs = rangefinder_inputs + random_inputs

            outputs = self.net.activate(inputs)
            self.last_inputs = inputs
            self.last_outputs = outputs

            if self.possessed:
                outputs = self.possession_outputs
                self.energy = MAX_ENERGY / 2

            thrust, turn, repr_threshold, repr_ratio = outputs[0], outputs[1], outputs[2], outputs[3]
            thrust = min(1.0, max(-1.0, thrust))
            turn = min(1.0, max(-1.0, turn))
            reproduction_energy_threshold = min(1.0, max(0, repr_threshold))
            reproduction_energy_ratio = min(1.0, max(0, repr_ratio))
            self.reproduction_energy_threshold = reproduction_energy_threshold * MAX_ENERGY
            self.reproduction_energy_ratio = reproduction_energy_ratio

            self.body.angle += turn * TURN_SPEED * dt

            direction = pymunk.Vec2d(1, 0)
            force = direction * thrust * AGENT_FORCE * dt
            self.body.apply_force_at_local_point(force)

        # energy costs
        self.energy -= ENERGY_DECAY * dt
        self.energy -= thrust ** 2 * ENERGY_THRUST * dt
        self.energy = min(self.energy, MAX_ENERGY)

        # should only spin through net control
        self.body.angular_velocity = 0

        # drag
        vel = self.body.velocity
        constant = -DRAG_CONST_COEFF * vel.normalized()
        linear = -DRAG_LINEAR_COEFF * vel
        quadratic = -DRAG_QUADRATIC_COEFF * vel * vel.length
        friction_force = constant + linear + quadratic
        self.body.apply_force_at_world_point(friction_force, self.body.position)

    def get_rangefinder_inputs(self, space):
        inputs = []
        angle_start = self.body.angle - RANGEFINDER_ANGLE / 2
        angle_step = RANGEFINDER_ANGLE / (N_RANGEFINDERS - 1)

        for i in range(N_RANGEFINDERS):
            angle = angle_start + i * angle_step
            ray_dir = pymunk.Vec2d(math.cos(angle), math.sin(angle))
            start = self.body.position
            end = start + ray_dir * RANGEFINDER_RADIUS

            hits = space.segment_query(start, end, 1, pymunk.ShapeFilter())

            min_distance = RANGEFINDER_RADIUS
            color = (0.0, 0.0, 0.0)

            for hit in hits:
                if hit.shape != self.shape:
                    hit_distance = hit.alpha * RANGEFINDER_RADIUS
                    if hit_distance < min_distance:
                        min_distance = hit_distance
                        color = hit.shape.owner.sight_color

            inputs.append(min_distance / RANGEFINDER_RADIUS)
            inputs.extend(color)

        return inputs

    def get_random_input(self):
        return [np.random.normal()]

    def reproduce(self):
        child_net = mutate_net(self.net)

        child_energy = self.energy * self.reproduction_energy_ratio
        self.energy -= child_energy

        facing = pymunk.Vec2d(math.cos(self.body.angle), math.sin(self.body.angle))
        child_angle = (self.body.angle + math.pi) % (2 * math.pi)
        spawn_position = self.body.position - facing * (AGENT_RADIUS * 4)

        return Agent(spawn_position, child_net, child_energy, child_angle)

    def draw_annotation(self, screen, env):
        # Draw selected agent's sensors and HUD
        pos = int(self.body.position.x), int(self.body.position.y)
        angle_start = self.body.angle - RANGEFINDER_ANGLE / 2
        angle_step = RANGEFINDER_ANGLE / (N_RANGEFINDERS - 1)
        for i in range(N_RANGEFINDERS):
            angle = angle_start + i * angle_step
            ray_dir = pymunk.Vec2d(math.cos(angle), math.sin(angle))
            start = self.body.position
            end_point = start + ray_dir * RANGEFINDER_RADIUS
            hits = env.space.segment_query(start, end_point, 1, pymunk.ShapeFilter())
            min_end = end_point
            for hit in hits:
                if hit.shape != self.shape:
                    min_end = start + ray_dir * (hit.alpha * RANGEFINDER_RADIUS)
                    break
            pygame.draw.line(screen, RAY_COLOR, pos, (int(min_end.x), int(min_end.y)), 1)

        # draw force
        scale = 1
        end = (pos[0] + self.body.velocity.x * scale,
               pos[1] + self.body.velocity.y * scale)
        pygame.draw.line(screen, (255, 255, 0), pos, end, 2)

        # Draw energy bar
        energy_ratio = self.energy / MAX_ENERGY
        pygame.draw.rect(screen, (255, 0, 0), (10, 10, 100, 10))
        pygame.draw.rect(screen, (0, 255, 0), (10, 10, int(100 * energy_ratio), 10))

        # Draw inputs and outputs
        font = pygame.font.SysFont(None, 24)
        y = 30
        for idx, val in enumerate(self.last_inputs):
            input_idx = self.net.input_ids[idx]
            input_node = self.net.node_map[input_idx]
            input_name = input_node.name

            text = font.render(f'I{idx} | {input_name}: {val:.2f}', True, (0, 255, 255))
            screen.blit(text, (10, y))
            y += 20
        for idx, val in enumerate(self.last_outputs):
            output_idx = self.net.output_ids[idx]
            output_node = self.net.node_map[output_idx]
            output_name = output_node.name

            text = font.render(f'O{idx} | {output_name}: {val:.2f}', True, (255, 0, 255))
            screen.blit(text, (10, y))
            y += 20
