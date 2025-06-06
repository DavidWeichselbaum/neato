# World
WIDTH = 1200
HEIGHT = 800
FPS_SIMULATION = 15

NUM_FOOD_INITIAL = 100
NUM_AGENTS_INITIAL = 10

# Agent
AGENT_RADIUS = 8
AGENT_FORCE = 5000
TURN_SPEED = 2.0
DRAG_CONST_COEFF = 5
DRAG_LINEAR_COEFF = 0.05
DRAG_QUADRATIC_COEFF = 0.005

N_RANGEFINDERS = 3
RANGEFINDER_RADIUS = 150
RANGEFINDER_ANGLE = 100

NETWORK_EVALUATION_STEP = 1
INPUT_NAMES = [[f"dist {i}", f"red {i}", f"green {i}", f"blue {i}"] for i in range(N_RANGEFINDERS)]
INPUT_NAMES = [name for group in INPUT_NAMES for name in group]
INPUT_NAMES += ["rand", "energy", "dir", "speed"]
N_INPUTS = len(INPUT_NAMES)
SPEED_INPUT_SCALE = 20
OUTPUT_NAMES = ["accel", "rot", "repro", "ratio"]  # accelerate, rotate, reproduction energy threshold, ratio of energy for offspring
N_OUTPUTS = len(OUTPUT_NAMES)

MAX_ENERGY = 100.0
ENERGY_DECAY = 3.0  # constant over time
ENERGY_REPRODUCTION = 10
ENERGY_THRUST = 10  # scales with the square of thrust

# Food
FOOD_SPAWN_INTERVAL = 0.1  # seconds simulation time
FOOD_RADIUS = 5
FOOD_MAX_AGE = 60  # seconds
FOOD_INITIAL_AGE = -90  # to stop mass extinction
FOOD_ENERGY = 50

# Walls
NUM_WALLS = 100
WALL_MIN_LEN = 50
WALL_MAX_LEN = 200
WALL_THICKNESS = 5
WALL_ELASTICITY = 0.8
WALL_FRICTION = 0.8

# Colors and colors seen by agents
AGENT_COLOR = (0, 0, 255)
AGENT_SIGHT_COLOR = (0.0, 0.0, 1.0)
FOOD_COLOR = (0, 255, 0)
FOOD_SIGHT_COLOR = (0.0, 1.0, 0.0)
WALL_COLOR = (255, 0, 0)
WALL_SIGHT_COLOR = (1.0, 0.0, 0.0)

# Colors
BACKGROUND_COLOR = (30, 30, 30)
FACING_COLOR = (0, 0, 255)
RAY_COLOR = (0, 255, 255)

# Misc
INFO_INTERVAL = 5  # seconds simulation time
FPS_VISUALIZATION = FPS_SIMULATION  # max visualization FPS
