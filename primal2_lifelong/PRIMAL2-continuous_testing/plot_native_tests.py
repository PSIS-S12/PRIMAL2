import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

BASELINE_RESULTS = "./results_50"
HEATMAP_RESULTS  = "./results_50_hm"

AGENTS = [4, 8, 16, 32, 64, 128]


def extract_agents(filename):
    match = re.search(r"(\d+)agents", filename)
    return int(match.group(1)) if match else None


def load_data(folder):
    throughput = defaultdict(list)
    planning = defaultdict(list)
    crashes = defaultdict(list)

    if not os.path.exists(folder):
        print(f"[WARN] Missing folder: {folder}")
        return {}, {}, {}

    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(folder, file)

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except:
            continue

        agents = extract_agents(file)
        if agents is None:
            continue

        times = data.get("computing time", [])
        target = data.get("target_reached", 0)
        num_crash = data.get("num_crash", 0)
        status = data.get("status", "")

        if not times:
            continue

        steps = len(times)

        # timeout handling
        if status == "timeout":
            throughput[agents].append(0)
            planning[agents].append(None)
            crashes[agents].append(num_crash)
            continue

        tp = target / steps
        pt = sum(times) / steps

        throughput[agents].append(tp)
        planning[agents].append(pt)
        crashes[agents].append(num_crash)

    tp_avg = {}
    pt_avg = {}
    crash_avg = {}

    for a in AGENTS:
        if throughput[a]:
            tp_avg[a] = sum(throughput[a]) / len(throughput[a])

        valid = [v for v in planning[a] if v is not None]
        if valid:
            pt_avg[a] = sum(valid) / len(valid)

        if crashes[a]:
            crash_avg[a] = sum(crashes[a]) / len(crashes[a])

    return tp_avg, pt_avg, crash_avg


def map_to_agents(data):
    return [data.get(a, None) for a in AGENTS]


def set_log_scale(ax):
    if matplotlib.__version__ < "3.3":
        ax.set_xscale("log", basex=2)
    else:
        ax.set_xscale("log", base=2)


def fix_ticks(ax):
    ax.set_xticks(AGENTS)
    ax.set_xticklabels([str(v) for v in AGENTS])


# -----------------------
# LOAD DATA
# -----------------------
tp_base, pt_base, crash_base = load_data(BASELINE_RESULTS)
tp_heat, pt_heat, crash_heat = load_data(HEATMAP_RESULTS)

x = AGENTS

y_base = map_to_agents(tp_base)
y_heat = map_to_agents(tp_heat)

p_base = map_to_agents(pt_base)
p_heat = map_to_agents(pt_heat)

c_base = map_to_agents(crash_base)
c_heat = map_to_agents(crash_heat)


# -----------------------
# PRINT TABLE
# -----------------------
print("\n=== RESULTS ===")
print(f"{'Agents':>8} | {'TP Base':>10} | {'TP Heat':>10} | {'Crash Base':>12} | {'Crash Heat':>12}")
print("-" * 70)

for a in AGENTS:
    print(
        f"{a:>8} | "
        f"{(round(tp_base.get(a, 0),3) if a in tp_base else '-'):>10} | "
        f"{(round(tp_heat.get(a, 0),3) if a in tp_heat else '-'):>10} | "
        f"{(round(crash_base.get(a, 0),3) if a in crash_base else '-'):>12} | "
        f"{(round(crash_heat.get(a, 0),3) if a in crash_heat else '-'):>12}"
    )


# -----------------------
# PLOT 1 — THROUGHPUT
# -----------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(x, y_base, marker="o", linewidth=2, markersize=8, label="Baseline")
ax1.plot(x, y_heat, marker="s", linewidth=2, markersize=8, label="Heatmap")

ax1.set_title("Throughput Comparison", fontsize=14, fontweight='bold')
ax1.set_xlabel("Team Size")
ax1.set_ylabel("Targets / timestep")

set_log_scale(ax1)
fix_ticks(ax1)

ax1.grid(True, alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.show()


# -----------------------
# PLOT 2 — PLANNING TIME
# -----------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(x, p_base, marker="o", linewidth=2, markersize=8, label="Baseline")
ax2.plot(x, p_heat, marker="s", linewidth=2, markersize=8, label="Heatmap")

ax2.set_title("Planning Time Comparison", fontsize=14, fontweight='bold')
ax2.set_xlabel("Team Size")
ax2.set_ylabel("Planning Time [s]")

set_log_scale(ax2)
ax2.set_yscale("log")
fix_ticks(ax2)

ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()


# -----------------------
# PLOT 3 — COLLISIONS
# -----------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.plot(x, c_base, marker="o", linewidth=2, markersize=8, label="Baseline")
ax3.plot(x, c_heat, marker="s", linewidth=2, markersize=8, label="Heatmap")

ax3.set_title("Collision Comparison (num_crash)", fontsize=14, fontweight='bold')
ax3.set_xlabel("Team Size")
ax3.set_ylabel("Average Collisions")

set_log_scale(ax3)
fix_ticks(ax3)

ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()