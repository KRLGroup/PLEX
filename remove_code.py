import glob
import shutil
import os

# Percorso base
base_path = "results/*/*/*/code"

# Trova tutte le cartelle corrispondenti
dirs_to_delete = glob.glob(base_path)

for d in dirs_to_delete:
    if os.path.isdir(d):
        print(f"Elimino: {d}")
        shutil.rmtree(d)

print("Eliminazione completata.")
