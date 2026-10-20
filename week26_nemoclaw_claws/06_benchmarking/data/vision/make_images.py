# Regenerate the three synthetic plant images used by tools/model_bakeoff.py (run from this folder).
"""Synthetic HVAC images with known answers, for vision tests."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
# 1. chiller plant table (kW/RT: CH-1 0.788, CH-2 0.638, CH-3 0.950 -> CH-3 worst)
fig, ax = plt.subplots(figsize=(7, 2.2)); ax.axis("off")
rows = [["CH-1", "410", "520", "6.8", "12.1"], ["CH-2", "300", "470", "6.9", "11.8"], ["CH-3", "380", "400", "7.4", "11.2"]]
t = ax.table(cellText=rows, colLabels=["Chiller", "Power (kW)", "Cooling load (RT)", "CHWS (°C)", "CHWR (°C)"], loc="center")
t.scale(1, 1.6); ax.set_title("Chiller plant — live values 14:05", fontsize=11)
fig.savefig("plant_table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
# 2. CHWS trend with a spike at 14:00 to 9.5 °C
h = np.arange(0, 24, 0.25); y = 6.7 + 0.08*np.sin(h/3.0)
y[(h >= 13.5) & (h <= 14.5)] = 6.7 + 2.8*np.exp(-((h[(h >= 13.5) & (h <= 14.5)] - 14.0)**2)/0.05)
fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(h, y); ax.set_xlabel("Hour of day"); ax.set_ylabel("CHWS temperature (°C)")
ax.set_xticks(range(0, 25, 2)); ax.grid(alpha=.3); ax.set_title("Chilled water supply temperature — 24 h")
fig.savefig("chws_trend.png", dpi=130, bbox_inches="tight"); plt.close(fig)
# 3. nameplate (rated kW/RT = 680/1200 = 0.567)
fig, ax = plt.subplots(figsize=(5, 2.6)); ax.axis("off")
ax.text(0.03, 0.95, "CENTRIFUGAL WATER CHILLER", fontsize=13, weight="bold", va="top")
for i, l in enumerate(["Model: AT-CH1200", "Nominal capacity: 1200 RT", "Refrigerant: R-134a",
                       "Rated input power: 680 kW", "Serial No.: AT-2026-0412", "Voltage: 400 V / 3 ph / 50 Hz"]):
    ax.text(0.03, 0.78 - i*0.13, l, fontsize=11, va="top", family="monospace")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=2, transform=ax.transAxes))
fig.savefig("nameplate.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("ok")
