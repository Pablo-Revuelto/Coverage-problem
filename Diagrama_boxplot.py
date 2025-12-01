import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# DATOS
# ==========================================

# GA: Genetic Algorithm
ga_data = {
    15: [7500.00, 709.08, 5500.00, 6500.00],
    20: [1010.52, 904.08, 839.28, 1114.09],
    25: [633.55, 830.86, 643.18, 498.34],
    30: [635.26, 580.16, 625.84, 648.47],
    35: [735.69, 690.14, 591.56, 743.16]
}

# SA: Simulated Annealing (Costes, la mayoría incluyen penalización)
sa_data = {
    15: [19355.03, 18387.47, 21899.71, 19862.42],
    20: [20393.39, 10976.49, 9495.80, 20945.88],
    25: [6580.13, 13530.14, 10560.80, 8589.53],
    30: [2203.44, 4214.41, 9176.59, 9145.25],
    35: [767.19, 8685.09, 7206.49, 7189.21]
}

# ==========================================
# PREPARACIÓN DE DATOS
# ==========================================
data_list = []

for k, values in ga_data.items():
    for val in values:
        data_list.append({"Waypoints (K)": k, "Cost (Fitness)": val, "Algorithm": "Proposed GA"})

for k, values in sa_data.items():
    for val in values:
        data_list.append({"Waypoints (K)": k, "Cost (Fitness)": val, "Algorithm": "SA"})

df = pd.DataFrame(data_list)

# ==========================================
# GENERACIÓN DEL PLOT
# ==========================================
# Configuración
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Crear el boxplot agrupado
# 'hue' separa los colores por algoritmo automáticamente
ax = sns.boxplot(data=df, x="Waypoints (K)", y="Cost (Fitness)", hue="Algorithm", 
                 palette=["#3498db", "#e74c3c"], linewidth=1.5, fliersize=5)

# 1. ESCALA LOGARÍTMICA: Fundamental por la diferencia de magnitudes
ax.set_yscale("log")

# 2. Títulos y etiquetas
ax.set_title("Cost Distribution Comparison: GA vs. SA", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Final Cost (Log Scale)", fontsize=12)
ax.set_xlabel("Number of Waypoints (K)", fontsize=12)

# 3. Leyenda
plt.legend(title="Method", title_fontsize=11, fontsize=10, loc='upper right', frameon=True, shadow=True)

# 4. Nota sobre la escala log
plt.text(0.02, 0.02, "Note: Y-axis is on a logarithmic scale due to high penalty values in SA results.", 
         transform=ax.transAxes, fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()

# Guardar imagen en alta resolución
output_file = "boxplot_comparison.png"
plt.savefig(output_file, dpi=300)
print(f"--> Gráfica generada correctamente: {output_file}")
plt.show()
