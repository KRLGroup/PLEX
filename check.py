from pathlib import Path

base_results = Path("/home/palu001/LogiX-Me/results")
base_logic = Path("/home/palu001/LogiX-Me/results_logic")

for f in sorted(base_results.glob("*/*")):
    if not f.is_dir():
        continue

    x = f.parent.name   # primo *
    y = f.name          # secondo *

    logic_root = base_logic / x
    if not logic_root.is_dir():
        continue

    for g in sorted(logic_root.glob(f"*/{y}")):
        if not g.is_dir():
            continue

        z = g.parent.name  # secondo * di results_logic

        print(
            "MATCH\n"
            f"  results       : {f}\n"
            f"    ├─ primo *  : {x}\n"
            f"    └─ secondo *: {y}\n"
            f"  results_logic : {g}\n"
            f"    ├─ primo *  : {x}\n"
            f"    └─ terzo *  : {y}\n"
            "----------------------------------------"
        )
