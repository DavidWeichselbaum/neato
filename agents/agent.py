import pymunk
import math
import random

from agents.constants import *


class Agent:
    def __init__(self, position, net, energy, angle):
        self.net = net
        self.energy = energy

        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, AGENT_RADIUS))
        self.body.position = position
        self.body.angle = angle
        # self.body.damping = 0.9

        self.shape = pymunk.Circle(self.body, AGENT_RADIUS)
        self.shape.elasticity = WALL_ELASTICITY
        self.shape.friction = WALL_FRICTION
        self.shape.owner = self

        self.color = AGENT_COLOR
        self.sight_color = AGENT_SIGHT_COLOR

        self.counter = 0
        self.last_inputs = []
        self.last_outputs = []

        self.possessed = False
        self.possession_outputs = [0.0, 0.0]

    def update(self, space):
        self.counter += 1
        if self.counter % NETWORK_EVALUATION_STEP == 0:

            inputs = self.get_rangefinder_inputs(space)
            outputs = self.net.activate(inputs)
            self.last_inputs = inputs
            self.last_outputs = outputs

            if self.possessed:
                outputs = self.possession_outputs
                self.energy += ENERGY_DECAY

            thrust = outputs[0]
            thrust = min(1.0, max(-1.0, thrust))
            turn = outputs[1]
            turn = min(1.0, max(-1.0, turn))

            self.body.angle += turn * TURN_SPEED * (1.0 / FPS)

            direction = pymunk.Vec2d(1, 0) * (1.0 / FPS)
            force = direction * thrust * AGENT_FORCE
            self.body.apply_force_at_local_point(force)

            self.energy -= ENERGY_DECAY

        self.body.angular_velocity = 0

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
