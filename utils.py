import os
import pathlib

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def create_folder(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def clear_and_create_folder(folder):
    create_folder(folder)
    clear_folder(folder)

def check_if_folder_exists(folder):
    return os.path.exists(folder)