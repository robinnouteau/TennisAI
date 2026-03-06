import os
import shutil

from loguru import logger


def makedirs_if_not_exist(path: str):
    if not os.path.exists(path): 
        # if the demo_folder directory is not present then create it. 
        os.makedirs(path)


def clean_dir(path: str, delete_subdir=True):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                if delete_subdir:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')