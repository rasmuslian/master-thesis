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

def prepare_folders():
    folders = [
        "output/train/increasing",
        "output/train/decreasing",
        "output/test/increasing",
        "output/test/decreasing",
        "output/validate/increasing",
        "output/validate/decreasing",
    ]

    for folder in folders:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        clear_folder(folder)
    