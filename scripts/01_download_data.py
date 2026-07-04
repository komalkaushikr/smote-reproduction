"""Stage 1: download the Pima Indian Diabetes dataset."""
import sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    if config.PIMA_CSV.exists():
        print(f"Already have {config.PIMA_CSV}")
        return
    print(f"Downloading Pima -> {config.PIMA_CSV}")
    urllib.request.urlretrieve(URL, config.PIMA_CSV)
    print("Done.")

if __name__ == "__main__":
    main()
