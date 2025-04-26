import pymunk
import math
import random

from agents.constants import *


class Agent:
    def __init__(self, space, position, net):
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, AGENT_RADIUS))
        self.body.position = position
        self.shape = pymunk.Circle(self.body, AGENT_RADIUS)
        self.shape.elasticity = WALL_ELASTICITY
        self.shape.friction = WALL_FRICTION
        space.add(self.body, self.shape)
        self.net = net
        self.counter = 0
        self.energy = INITIAL_ENERGY
        self.facing_angle = random.uniform(0, 2 * math.pi)
        self.last_inputs = []
        self.last_outputs = []

    def update(self, space):
        self.counter += 1
        if self.counter % NETWORK_EVALUATION_STEP == 0:
            inputs = self.get_rangefinder_inputs(space)
            outputs = self.net.activate(inputs)
            self.last_inputs = inputs
            self.last_outputs = outputs
            thrust = outputs[0]
            turn = outputs[1] * 2 - 1

            self.facing_angle += turn * TURN_SPEED * (1.0 / FPS)

            # Cancel lateral drift
            forward = pymunk.Vec2d(math.cos(self.facing_angle), math.sin(self.facing_angle))
            forward_speed = forward.dot(self.body.velocity)
            self.body.velocity = forward * forward_speed

            self.body.apply_force_at_local_point(forward * thrust * AGENT_FORCE)
            self.energy -= ENERGY_DECAY

    def get_rangefinder_inputs(self, space):
        inputs = []
        angle_start = self.facing_angle - math.pi / 2
        angle_step = math.pi / (N_RANGEFINDERS - 1)
        for i in range(N_RANGEFINDERS):
            angle = angle_start + i * angle_step
            ray_dir = pymunk.Vec2d(math.cos(angle), math.sin(angle))
            start = self.body.position
            end = start + ray_dir * RANGEFINDER_RADIUS
            hits = space.segment_query(start, end, 1, pymunk.ShapeFilter())
            min_distance = RANGEFINDER_RADIUS
            for hit in hits:
                if hit.shape != self.shape:
                    min_distance = min(min_distance, hit.alpha * RANGEFINDER_RADIUS)
            inputs.append(min_distance / RANGEFINDER_RADIUS)
        return inputs
