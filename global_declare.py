
# Image dimensions
IN_HEIGHT = 448
IN_WIDTH = 448
CHANNEL = 3
IMAGE_SIZE = 448

# cell grid size
GRID_SIZE = 28
NO_BOUNDING_BOX = 2
NO_CLASSES = 9
TOTAL_OUTPUTS = GRID_SIZE * GRID_SIZE * (NO_CLASSES + NO_BOUNDING_BOX * 5)

# 
ALPHA = 0.1
CLASS_SCALE= 2.0
NOOBJ_SCALE = 1.0
OBJ_SCALE = 2.0
COORD_SCALE = 5.0

PROB_THRESHOLD = 0.2
IOU_THRESHOLD = 0.4

CLASSES = ['car', 'truck', 'tractor', 'campingcar', 'van', 'pickup' , 'boat', 'plane', 'other']
