from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "progress_data.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("LEARNING GROWTH ANALYTICS")
print("=" * 48)
print(df.to_string(index=False))

subjects = ["math", "reading"]
summary_rows = []
for subject in subjects:
    first = df.iloc[0][f"{subject}_rit"]
    spring = df.loc[df["term"] == "Spring 2025", f"{subject}_rit"].iloc[0]
    fall = df.loc[df["term"] == "Fall 2025", f"{subject}_rit"].iloc[0]
    total_gain = fall - first
    spring_to_fall = fall - spring

    spring_pct = df.loc[df["term"] == "Spring 2025", f"{subject}_achievement_percentile"].iloc[0]
    fall_pct = df.loc[df["term"] == "Fall 2025", f"{subject}_achievement_percentile"].iloc[0]
    percentile_gain = fall_pct - spring_pct

    summary_rows.append({
        "subject": subject.title(),
        "RIT gain Fall 2024 -> Fall 2025": int(total_gain),
        "RIT gain Spring 2025 -> Fall 2025": int(spring_to_fall),
        "Achievement percentile gain Spring -> Fall": int(percentile_gain),
    })

summary = pd.DataFrame(summary_rows)
print("\nSUMMARY")
print("=" * 48)
print(summary.to_string(index=False))
summary.to_csv(OUT / "growth_summary.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(df["term"], df["math_rit"], marker="o", label="Mathematics")
plt.plot(df["term"], df["reading_rit"], marker="o", label="Reading")
plt.title("Learning Growth Across Four Assessment Periods")
plt.xlabel("Assessment period")
plt.ylabel("RIT score")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "rit_growth.png", dpi=180)
plt.close()

pct = df[df["math_achievement_percentile"].notna()].copy()
plt.figure(figsize=(8, 5))
plt.plot(pct["term"], pct["math_achievement_percentile"], marker="o", label="Mathematics")
plt.plot(pct["term"], pct["reading_achievement_percentile"], marker="o", label="Reading")
plt.title("Achievement Percentile Change: Spring to Fall 2025")
plt.xlabel("Assessment period")
plt.ylabel("Achievement percentile")
plt.ylim(0, 100)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "achievement_percentiles.png", dpi=180)
plt.close()

print("\nCreated:")
print(" - output/growth_summary.csv")
print(" - output/rit_growth.png")
print(" - output/achievement_percentiles.png")
print("\nInterpretation:")
print("Mathematics shows the strongest measurable change, including a 23-point")
print("achievement-percentile increase from Spring 2025 to Fall 2025.")
