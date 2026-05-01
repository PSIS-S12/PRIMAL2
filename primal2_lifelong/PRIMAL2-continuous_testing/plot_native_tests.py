
import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib


RESULTS_40 = "./results_40_dense"
RESULTS_160 = "./results_160_sparse"

AGENTS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def extract_agents(filename):
    match = re.search(r"(\d+)agents", filename)
    return int(match.group(1)) if match else None


def load_data(folder):
    throughput = defaultdict(list)
    planning = defaultdict(list)

    if not os.path.exists(folder):
        print(f"[WARN] Missing folder: {folder}")
        return {}, {}

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
        status = data.get("status", "")

        if not times:
            continue

        steps = len(times)

        # include timeouts as zero throughput
        if status == "timeout":
            throughput[agents].append(0)
            planning[agents].append(None)
            continue

        tp = target / steps
        pt = sum(times) / steps

        throughput[agents].append(tp)
        planning[agents].append(pt)

    # average
    tp_avg = {}
    pt_avg = {}

    for a in AGENTS:
        if a in throughput and throughput[a]:
            tp_avg[a] = sum(throughput[a]) / len(throughput[a])

        if a in planning:
            valid = [v for v in planning[a] if v is not None]
            if valid:
                pt_avg[a] = sum(valid) / len(valid)

    return tp_avg, pt_avg


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


tp40, _ = load_data(RESULTS_40)
tp160, pt160 = load_data(RESULTS_160)

x = AGENTS
y40 = map_to_agents(tp40)
y160 = map_to_agents(tp160)
yp = map_to_agents(pt160)


# -----------------------
# PRINT TABLE
# -----------------------
print("\n=== RESULTS ===")
print(f"{'Agents':>8} | {'TP40':>8} | {'TP160':>8} | {'Plan[s]':>8}")
print("-" * 40)

for a in AGENTS:
    print(f"{a:>8} | "
          f"{(round(tp40.get(a, 0),3) if a in tp40 else '-'):>8} | "
          f"{(round(tp160.get(a, 0),3) if a in tp160 else '-'):>8} | "
          f"{(round(pt160.get(a, 0),3) if a in pt160 else '-'):>8}")


# -----------------------
# PLOT 1 — Throughput 40x40
# -----------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x, y40, marker="o", linewidth=2, markersize=8)
ax1.set_title("Throughput - 40x40", fontsize=14, fontweight='bold')
ax1.set_xlabel("Team Size", fontsize=12)
ax1.set_ylabel("Targets / timestep", fontsize=12)
set_log_scale(ax1)
fix_ticks(ax1)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------
# PLOT 2 — Throughput 160x160
# -----------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x, y160, marker="o", linewidth=2, markersize=8)
ax2.set_title("Throughput - 160x160", fontsize=14, fontweight='bold')
ax2.set_xlabel("Team Size", fontsize=12)
ax2.set_ylabel("Targets / timestep", fontsize=12)
set_log_scale(ax2)
fix_ticks(ax2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------
# PLOT 3 — Planning Time
# -----------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(x, yp, marker="o", linewidth=2, markersize=8, color='orange')
ax3.set_title("Planning Time", fontsize=14, fontweight='bold')
ax3.set_xlabel("Team Size", fontsize=12)
ax3.set_ylabel("Time [s]", fontsize=12)
set_log_scale(ax3)
ax3.set_yscale("log")
fix_ticks(ax3)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -----------------------
# PLOT 4 — PRIMAL2 Throughput
# -----------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(x, y160, marker="o", linewidth=2, markersize=8, color='green', label="PRIMAL2")
ax4.set_title("Throughput (PRIMAL2)", fontsize=14, fontweight='bold')
ax4.set_xlabel("Team Size", fontsize=12)
ax4.set_ylabel("Targets / timestep", fontsize=12)
set_log_scale(ax4)
fix_ticks(ax4)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11)
plt.tight_layout()
plt.show()