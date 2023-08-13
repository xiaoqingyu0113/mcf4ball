from pathlib import Path
import csv


CURRENT_DIR = Path(__file__)

def write_ignores(ignores):
    for folder_name, ignore_indices in ignores.items():
        with open(CURRENT_DIR/folder_name/'d_ignores.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(ignore_indices)


if __name__ == '__main__':
    ignores = {'tennis_1':[]}