import glob
import json

import glob
import json
import os

# prende tutti i dataset
base_path = "/home/palu001/LogiX-Me/results/*/*/results.json"
files = glob.glob(base_path)

for file_path in files:
    try:
        # Estrae il nome del dataset dal path
        parts = file_path.split(os.sep)
        dataset_name = parts[5]  # 0:/, 1:home, 2:palu001, 3:LogiX-Me, 4:results_logic, 5:<dataset_name>

        print(f"\n=== Dataset: {dataset_name} ===")

        with open(file_path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Errore nel file {file_path}: {e}")

