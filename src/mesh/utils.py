from pathlib import Path

def get_model_dir():
    return (Path(__file__).parent / "../../models")\
            .resolve().absolute()

def get_data_dir():
    return (Path(__file__).parent / "../../data")\
            .resolve().absolute()