# paths
IMAGE_PATH = '/scratch/ramrao/vehicles/train.txt'
LABEL_PATH = '/scratch/ramrao/vehicles/label.txt'

# Image dimensions
IN_HEIGHT = 448
IN_WIDTH = 448
CHANNEL = 3
IMAGE_SIZE = 448

# cell grid size
GRID_SIZE = 7
NO_BOUNDING_BOX = 2
NO_CLASSES = 9

# 
ALPHA = 0.1
INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True 

CLASS_SCALE= 2.0
NOOBJ_SCALE = 1.0
OBJ_SCALE = 2.0
COORD_SCALE = 5.0
