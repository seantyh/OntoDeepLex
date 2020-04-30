from pathlib import Path

def get_model_dir():
    return (Path(__file__).parent / "../../models")\
            .resolve().absolute()

def get_data_dir():
    return (Path(__file__).parent / "../../data")\
            .resolve().absolute()

def ensure_dir(path: Path, path_is_dir=True):
    if not path_is_dir:
        dir_path = path.parent
    else:
        dir_path = path
    dir_path.mkdir(parents=True, exist_ok=True)
