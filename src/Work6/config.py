from pathlib import Path

# =========================
# Path Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSET_DIR = PROJECT_ROOT / "assets" / "Work6"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Work6"

TARGET_MESH_PATH = ASSET_DIR / "cow.obj"

SAVE_DIR = OUTPUT_DIR
IMAGE_DIR = SAVE_DIR / "images"
MESH_DIR = SAVE_DIR / "output_meshes"


# =========================
# Device Configuration
# =========================

DEVICE = "cuda:0"


# =========================
# Rendering Configuration
# =========================

IMAGE_SIZE = 256

NUM_VIEWS = 20
CAMERA_DISTANCE = 2.7
CAMERA_ELEVATION = 0.0
CAMERA_AZIM_START = -180.0
CAMERA_AZIM_END = 180.0

SIGMA = 1e-4
GAMMA = 1e-4
FACES_PER_PIXEL = 50


# =========================
# Source Mesh Configuration
# =========================

SOURCE_SPHERE_LEVEL = 4


# =========================
# Optimization Configuration
# =========================

EPOCHS = 300
LEARNING_RATE = 1.0
MOMENTUM = 0.9

LAPLACIAN_WEIGHT = 1.0
EDGE_WEIGHT = 0.1
NORMAL_WEIGHT = 0.01


# =========================
# Output Configuration
# =========================

DISPLAY_INTERVAL = 20

SAVE_PROGRESS_IMAGES = True
SAVE_LOSS_CURVE = True
MAKE_GIF = True
GIF_NAME = "work6_optimization.gif"

RANDOM_SEED = 42
