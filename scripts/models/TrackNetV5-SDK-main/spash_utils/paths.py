from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def get_root_dir():
    return ROOT_DIR


def get_inference_dir():
    return ROOT_DIR / 'inference'


def get_dataset_dir():
    return ROOT_DIR / 'datasets'


def get_resources_dir():
    return get_inference_dir() / 'resources'


def get_images_dir():
    return get_resources_dir() / 'images'


def get_court_background_image_path():
    return get_images_dir() / 'padel1.png'


def get_padel_dataset_dir():
    return get_dataset_dir() / "padel"


def get_test_esprit_dir():
    return get_padel_dataset_dir() / "TestEsprit"


def get_test_paris_dir():
    return get_padel_dataset_dir() / "TestParis"


def get_S3_dir():
    return get_padel_dataset_dir() / "S3"


def get_new_captation_dir():
    return get_dataset_dir() / "New_captation"


def get_sessions_dir():
    return get_padel_dataset_dir() / "Sessions"


def get_stats_dir():
    """
    Directory of stats.json sessions per center
    """
    return get_dataset_dir() / "stats"


def get_models_dir():
    return get_root_dir() / "models"


DATASET_DIR = get_dataset_dir()
PADEL_DATASET_DIR = get_padel_dataset_dir()
TEST_ESPRIT_DIR = get_test_esprit_dir()
TEST_PARIS_DIR = get_test_paris_dir()
S3_DIR = get_S3_dir()
NEW_CAPTATION_DIR = get_new_captation_dir()
SESSIONS_DIR = get_sessions_dir()
MODELS_DIR = get_models_dir()
STATS_DIR = get_stats_dir()
