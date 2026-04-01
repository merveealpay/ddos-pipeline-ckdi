import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Girdi CSV
csv_path = Path("results_outputs/sample_level_ckdi_bdri.csv")

# Çıktı PDF
out_path = Path("figures/ckdi_vs_bdri_scatter.pdf")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Veriyi oku
df = pd.read_csv(csv_path)

# Grafik
plt.figure(figsize=(6, 4))
plt.scatter(df["CKDI"], df["BDRI_opt"], s=20, alpha=0.8)

plt.xlabel("CKDI")
plt.ylabel("BDRI (optimized)")
plt.title("CKDI vs Optimized BDRI")
plt.tight_layout()

# Kaydet
plt.savefig(out_path)
plt.close()

print(f"Saved: {out_path}")