import argparse
import json
from collections import defaultdict
from pathlib import Path


PLANNER_LABELS = {
    "testing_result_64": "PRIMAL2",
    "testing_result_512": "PRIMAL2",
    "testing_result_2048": "PRIMAL2",
}


# ------------------------
# UTIL
# ------------------------

def load_result_file(path):
    with path.open("r") as f:
        return json.load(f)


def infer_category(name):
    name_low = name.lower()
    if name_low.startswith("maze"):
        return "Maze"
    if name_low.startswith("warehouse"):
        return "Warehouse"
    return None


def get_agent_bucket(folder_name):
    if "64" in folder_name:
        return "4-64"
    elif "512" in folder_name:
        return "128-512"
    elif "2048" in folder_name:
        return "1024-2048"
    return None


# ------------------------
# LOAD RESULTS
# ------------------------

def collect_results(result_dir):
    result_dir = Path(result_dir)
    rows = []

    print(f"\n[INFO] Scanning: {result_dir}")

    for file_path in sorted(result_dir.glob("*_continuousRL.txt")):
        data = load_result_file(file_path)

        name = file_path.stem.replace("_continuousRL", "")
        category = infer_category(name)

        times = data.get("computing time", []) or []
        total_time = float(sum(times)) if times else 0.0
        target = int(data.get("target_reached", 0))

        throughput = None
        if total_time > 0:
            throughput = target / total_time

        print(f"[DEBUG] {name} | cat={category} | throughput={throughput}")

        rows.append({
            "name": name,
            "category": category,
            "throughput": throughput,
        })

    return rows


# ------------------------
# TABLE GENERATION
# ------------------------

def generate_table(result_dirs):
    aggregated = defaultdict(lambda: defaultdict(list))

    for result_dir in result_dirs:
        folder_name = Path(result_dir).name
        planner = PLANNER_LABELS.get(folder_name, "PRIMAL2")

        bucket = get_agent_bucket(folder_name)
        print(f"\n[INFO] Folder {folder_name} → bucket {bucket}")

        rows = collect_results(result_dir)

        for r in rows:
            if r["throughput"] is None:
                print(f"[SKIP] {r['name']} → no throughput")
                continue

            if r["category"] not in ["Maze", "Warehouse"]:
                print(f"[SKIP] {r['name']} → unknown category")
                continue

            key = (r["category"], bucket)
            aggregated[planner][key].append(r["throughput"])

            print(f"[AGG] {planner} → {key} += {r['throughput']:.4f}")

    # compute averages
    avg = defaultdict(dict)
    for planner, data in aggregated.items():
        for key, values in data.items():
            avg_val = sum(values) / len(values)
            avg[planner][key] = avg_val
            print(f"[AVG] {planner} → {key} = {avg_val:.4f} ({len(values)} samples)")

    # columns in desired order
    columns = [
        ("Maze", "4-64"),
        ("Maze", "128-512"),
        ("Maze", "1024-2048"),
        ("Warehouse", "4-64"),
        ("Warehouse", "128-512"),
        ("Warehouse", "1024-2048"),
    ]

    # print table
    col_width = 20

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.2f}"
        return str(x)

    header = ["Planner"] + [f"{c[0]} ({c[1]})" for c in columns]
    print("\n" + " ".join(h.ljust(col_width) for h in header))

    for planner in sorted(avg.keys()):
        row = [planner]
        for col in columns:
            value = avg[planner].get(col, "-")
            row.append(fmt(value))
        print(" ".join(str(x).ljust(col_width) for x in row))


# ------------------------
# MAIN
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        default=[
            "/home/nik/PycharmProjects/PRIMAL2-psis/primal2_lifelong/PRIMAL2-continuous_testing/testing_result_64",
            "/home/nik/PycharmProjects/PRIMAL2-psis/primal2_lifelong/PRIMAL2-continuous_testing/testing_result_512",
            "/home/nik/PycharmProjects/PRIMAL2-psis/primal2_lifelong/PRIMAL2-continuous_testing/testing_result_2048",
        ],
    )

    args = parser.parse_args()
    generate_table(args.result_dirs)


if __name__ == "__main__":
    main()