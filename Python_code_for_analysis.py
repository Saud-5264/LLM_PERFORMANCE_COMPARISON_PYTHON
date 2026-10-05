"""
================================================================================
COMPREHENSIVE LLM BENCHMARK COMPARISON & MULTI-DIMENSIONAL ANALYSIS
================================================================================
Required Libraries:
  - numpy: Vectorized metrics, normalization, Pareto frontier calculation, pure NumPy KDE
  - pandas: Data ingestion, cleaning, multi-level aggregations, ranking, multi-sheet export
  - matplotlib: Multi-panel executive dashboards, Pareto frontier curves, Green AI charts
  - seaborn: Correlation heatmaps, provider benchmark distributions, violin plots, joint distribution
  - plotly express: 3D interactive trade-off explorer, bubble charts, sunburst hierarchy,
                    parallel coordinates SLA filter, and grouped comparison bar charts
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for clean automated plot generation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# 1. Environment & Styling Configuration
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILENAME = 'llm_comparison_dataset.xlsx'
EXCEL_PATH = os.path.join(SCRIPT_DIR, EXCEL_FILENAME)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seaborn & Matplotlib Global Theme Settings
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams.update({
    'figure.autolayout': True,
    'figure.titlesize': 15,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Curated provider color palette for consistent visual identification
PROVIDER_COLORS = {
    'OpenAI': '#10a37f',
    'Anthropic': '#cc785c',
    'Google': '#4285f4',
    'Meta AI': '#0668e1',
    'Deepseek': '#0284c7',
    'Mistral AI': '#ea580c',
    'Cohere': '#4f46e5',
    'AWS': '#d97706'
}

print("=" * 85)
print("          ENTERPRISE LLM COMPARATIVE BENCHMARK & PERFORMANCE ANALYSIS")
print("=" * 85)
print(f"[*] Python Runtime      : {sys.version.split()[0]}")
print(f"[*] NumPy Version       : {np.__version__}")
print(f"[*] Pandas Version      : {pd.__version__}")
print(f"[*] Matplotlib Version  : {plt.matplotlib.__version__}")
print(f"[*] Seaborn Version     : {sns.__version__}")
print(f"[*] Target Directory    : {SCRIPT_DIR}")
print(f"[*] Source Excel File   : {EXCEL_PATH}")
print(f"[*] Visualizations Dir  : {OUTPUT_DIR}")
print("=" * 85)


# ------------------------------------------------------------------------------
# 2. Pure NumPy Helper Functions
# ------------------------------------------------------------------------------
def numpy_kde(data, grid, bandwidth=None):
    """
    Computes Gaussian Kernel Density Estimation (KDE) using pure NumPy.
    Avoids external dependencies like scipy while delivering smooth probability densities.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if bandwidth is None:
        # Silverman's rule of thumb for bandwidth selection
        std_val = np.std(data, ddof=1)
        bandwidth = 1.06 * (std_val if std_val > 0 else 1.0) * (n ** (-0.2))
    bandwidth = max(bandwidth, 1e-4)
    u = (grid[:, None] - data[None, :]) / bandwidth
    density = np.sum(np.exp(-0.5 * (u ** 2)) / np.sqrt(2 * np.pi), axis=1) / (n * bandwidth)
    return density


def compute_pareto_frontier(x, y, x_min=True, y_min=False):
    """
    Identifies Pareto-optimal (non-dominated) points across 2 dimensions using NumPy.
    - x_min=True  => Lower X is better (e.g. Price, Latency)
    - y_min=False => Higher Y is better (e.g. Benchmark Elo, Speed)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] <= x[i]) if x_min else (x[j] >= x[i])
            y_better = (y[j] >= y[i]) if not y_min else (y[j] <= y[i])
            x_strict = (x[j] < x[i]) if x_min else (x[j] > x[i])
            y_strict = (y[j] > y[i]) if not y_min else (y[j] < y[i])
            if x_better and y_better and (x_strict or y_strict):
                is_pareto[i] = False
                break
    return is_pareto


# ------------------------------------------------------------------------------
# 3. Data Ingestion & Feature Engineering (Pandas & NumPy)
# ------------------------------------------------------------------------------
def load_and_enrich_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel dataset not found at: {file_path}")

    # Load first sheet of the workbook
    df = pd.read_excel(file_path, sheet_name=0)
    print(f"\n[+] Successfully ingested dataset: {df.shape[0]} records x {df.shape[1]} features.")

    # Integrity verification
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[!] Warning: Missing values detected:\n{null_counts[null_counts > 0]}")
        df = df.dropna().copy()
    else:
        print("[+] Data validation passed: 100% complete records (0 missing values).")

    # 1. Human-readable Open-Source category label
    df['Open_Source_Label'] = np.where(df['Open-Source'] == 1, 'Open Source', 'Proprietary')

    # 2. Responsiveness Efficiency (Throughput-to-Latency Ratio)
    # Reflects how many tokens/sec the model delivers relative to its initial wait time
    df['Throughput_Latency_Ratio'] = np.round(df['Speed (tokens/sec)'] / df['Latency (sec)'], 2)

    # 3. Cost-Efficiency Ratio (Chatbot Arena Elo points per dollar spent)
    df['Cost_Performance_Ratio'] = np.round(df['Benchmark (Chatbot Arena)'] / df['Price / Million Tokens'], 2)

    # 4. Academic Reasoning per dollar (MMLU score per dollar)
    df['MMLU_Per_Dollar'] = np.round(df['Benchmark (MMLU)'] / df['Price / Million Tokens'], 2)

    # 5. Multi-Criteria Composite Score (0 - 100 scale using NumPy Min-Max Normalization)
    def min_max_normalize(series, invert=False):
        arr = series.to_numpy(dtype=float)
        min_v = np.min(arr)
        max_v = np.max(arr)
        if max_v == min_v:
            return np.ones_like(arr) * 0.5
        norm = (arr - min_v) / (max_v - min_v)
        return (1.0 - norm) if invert else norm

    norm_arena = min_max_normalize(df['Benchmark (Chatbot Arena)'])
    norm_mmlu = min_max_normalize(df['Benchmark (MMLU)'])
    norm_speed = min_max_normalize(df['Speed (tokens/sec)'])
    norm_latency = min_max_normalize(df['Latency (sec)'], invert=True)
    norm_price = min_max_normalize(df['Price / Million Tokens'], invert=True)
    norm_energy = min_max_normalize(df['Energy Efficiency'])

    # Weighted Composite Score Breakdown:
    # 25% Chatbot Arena Elo (Empirical user preference)
    # 15% MMLU Benchmark (Broad multi-task reasoning)
    # 20% Generation Throughput (Speed in tokens/sec)
    # 15% Response Latency (First token delay, inverted)
    # 15% Token Pricing (Cost per million tokens, inverted)
    # 10% Energy Efficiency (Sustainability metric)
    composite = (
        0.25 * norm_arena +
        0.15 * norm_mmlu +
        0.20 * norm_speed +
        0.15 * norm_latency +
        0.15 * norm_price +
        0.10 * norm_energy
    ) * 100.0
    df['Composite_Score'] = np.round(composite, 2)

    # 6. Compute Pareto Frontiers
    price_vals = df['Price / Million Tokens'].to_numpy()
    arena_vals = df['Benchmark (Chatbot Arena)'].to_numpy()
    speed_vals = df['Speed (tokens/sec)'].to_numpy()
    latency_vals = df['Latency (sec)'].to_numpy()

    df['Is_Cost_Pareto'] = compute_pareto_frontier(price_vals, arena_vals, x_min=True, y_min=False)
    df['Is_Speed_Pareto'] = compute_pareto_frontier(latency_vals, speed_vals, x_min=True, y_min=False)

    return df

df = load_and_enrich_data(EXCEL_PATH)


# ------------------------------------------------------------------------------
# 4. Exploratory Data Analysis & Statistical Aggregations (Pandas & NumPy)
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("                           KEY STATISTICAL SUMMARY")
print("=" * 85)

numerical_cols = [
    'Context Window', 'Speed (tokens/sec)', 'Latency (sec)',
    'Benchmark (MMLU)', 'Benchmark (Chatbot Arena)', 'Price / Million Tokens',
    'Training Dataset Size', 'Compute Power', 'Energy Efficiency',
    'Throughput_Latency_Ratio', 'Cost_Performance_Ratio', 'Composite_Score'
]

summary_stats = df[numerical_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print(summary_stats.round(2).to_string())

# Provider Benchmark Matrix
print("\n" + "-" * 85)
print("                     PROVIDER-LEVEL PERFORMANCE BENCHMARKS")
print("-" * 85)
provider_summary = df.groupby('Provider').agg({
    'Model': 'count',
    'Benchmark (Chatbot Arena)': ['mean', 'median', 'std'],
    'Benchmark (MMLU)': ['mean', 'median'],
    'Speed (tokens/sec)': 'mean',
    'Latency (sec)': 'mean',
    'Price / Million Tokens': 'mean',
    'Composite_Score': 'mean'
}).round(2)
provider_summary.columns = [
    'Model_Count', 'Arena_Mean', 'Arena_Median', 'Arena_Std',
    'MMLU_Mean', 'MMLU_Median', 'Speed_Mean', 'Latency_Mean',
    'Price_Mean', 'Composite_Score_Mean'
]
provider_summary = provider_summary.sort_values('Composite_Score_Mean', ascending=False)
print(provider_summary.to_string())

# Open-Source vs Proprietary Architecture Breakdown
print("\n" + "-" * 85)
print("                 OPEN-SOURCE VS PROPRIETARY ECOSYSTEM COMPARISON")
print("-" * 85)
os_summary = df.groupby('Open_Source_Label').agg({
    'Model': 'count',
    'Benchmark (Chatbot Arena)': ['mean', 'std'],
    'Benchmark (MMLU)': ['mean', 'std'],
    'Speed (tokens/sec)': 'mean',
    'Latency (sec)': 'mean',
    'Price / Million Tokens': 'mean',
    'Composite_Score': 'mean'
}).round(2)
os_summary.columns = [
    'Model_Count', 'Arena_Mean', 'Arena_Std', 'MMLU_Mean',
    'MMLU_Std', 'Speed_Mean', 'Latency_Mean', 'Price_Mean', 'Composite_Score_Mean'
]
print(os_summary.to_string())

# Top 10 Overall Composite Leaders
print("\n" + "-" * 85)
print("                TOP 10 OVERALL MODELS (BALANCED COMPOSITE SCORE)")
print("-" * 85)
top_composite = df.sort_values('Composite_Score', ascending=False).head(10)[
    ['Model', 'Provider', 'Open_Source_Label', 'Benchmark (Chatbot Arena)',
     'Benchmark (MMLU)', 'Speed (tokens/sec)', 'Latency (sec)',
     'Price / Million Tokens', 'Composite_Score']
]
print(top_composite.to_string(index=False))

# Top 10 Value Champions (Arena Points per Dollar)
print("\n" + "-" * 85)
print("             TOP 10 BEST VALUE MODELS (CHATBOT ARENA ELO PER $)")
print("-" * 85)
top_value = df.sort_values('Cost_Performance_Ratio', ascending=False).head(10)[
    ['Model', 'Provider', 'Open_Source_Label', 'Price / Million Tokens',
     'Benchmark (Chatbot Arena)', 'Cost_Performance_Ratio']
]
print(top_value.to_string(index=False))

# Cost vs Quality Pareto-Optimal Frontier Models
print("\n" + "-" * 85)
print("       PARETO OPTIMAL FRONTIER: QUALITY (ARENA ELO) VS COST ($/M TOKENS)")
print("-" * 85)
pareto_cost_df = df[df['Is_Cost_Pareto']].sort_values('Price / Million Tokens')[
    ['Model', 'Provider', 'Open_Source_Label', 'Price / Million Tokens',
     'Benchmark (Chatbot Arena)', 'Benchmark (MMLU)', 'Speed (tokens/sec)', 'Latency (sec)']
]
print(pareto_cost_df.to_string(index=False))

# Export Comprehensive Summary to Multi-Sheet Excel Workbook
summary_excel_path = os.path.join(OUTPUT_DIR, 'llm_analysis_summary.xlsx')
with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as writer:
    summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
    provider_summary.to_excel(writer, sheet_name='Provider_Comparison')
    os_summary.to_excel(writer, sheet_name='Open_vs_Proprietary')
    top_composite.to_excel(writer, sheet_name='Top_Composite_Models', index=False)
    top_value.to_excel(writer, sheet_name='Top_Value_Models', index=False)
    pareto_cost_df.to_excel(writer, sheet_name='Pareto_Cost_Frontier', index=False)
    df.to_excel(writer, sheet_name='Enriched_Full_Dataset', index=False)
print(f"\n[+] Saved multi-sheet analytical Excel workbook: {summary_excel_path}")


# ------------------------------------------------------------------------------
# 5. Matplotlib Visualizations (Publication-Quality Static Plots)
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("                   GENERATING MATPLOTLIB VISUALIZATIONS")
print("=" * 85)

# ------------------------------------------------------------------------------
# Figure 1: Executive Overview Multi-Panel Dashboard
# ------------------------------------------------------------------------------
fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('LLM Comparative Performance: Executive Overview Dashboard', fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Distribution of Chatbot Arena Elo Scores
ax1 = axes[0, 0]
arena_data = df['Benchmark (Chatbot Arena)'].to_numpy()
ax1.hist(arena_data, bins=20, color='#2563eb', edgecolor='black', alpha=0.7, density=True)
kde_x_arena = np.linspace(arena_data.min(), arena_data.max(), 200)
kde_y_arena = numpy_kde(arena_data, kde_x_arena)
ax1.plot(kde_x_arena, kde_y_arena, color='#1e3a8a', linewidth=2.5, label='KDE Density')
ax1.axvline(np.mean(arena_data), color='#dc2626', linestyle='--', linewidth=2, label=f'Mean: {np.mean(arena_data):.1f}')
ax1.axvline(np.median(arena_data), color='#16a34a', linestyle=':', linewidth=2, label=f'Median: {np.median(arena_data):.1f}')
ax1.set_title('A. Chatbot Arena Elo Score Distribution', fontweight='bold')
ax1.set_xlabel('Chatbot Arena Elo')
ax1.set_ylabel('Density')
ax1.legend(loc='upper right')

# Subplot 2: Distribution of MMLU Benchmark Scores
ax2 = axes[0, 1]
mmlu_data = df['Benchmark (MMLU)'].to_numpy()
ax2.hist(mmlu_data, bins=15, color='#8b5cf6', edgecolor='black', alpha=0.7, density=True)
kde_x_mmlu = np.linspace(mmlu_data.min(), mmlu_data.max(), 200)
kde_y_mmlu = numpy_kde(mmlu_data, kde_x_mmlu)
ax2.plot(kde_x_mmlu, kde_y_mmlu, color='#5b21b6', linewidth=2.5, label='KDE Density')
ax2.axvline(np.mean(mmlu_data), color='#dc2626', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mmlu_data):.1f}')
ax2.axvline(np.median(mmlu_data), color='#16a34a', linestyle=':', linewidth=2, label=f'Median: {np.median(mmlu_data):.1f}')
ax2.set_title('B. MMLU Benchmark Score Distribution', fontweight='bold')
ax2.set_xlabel('MMLU Score (%)')
ax2.set_ylabel('Density')
ax2.legend(loc='upper right')

# Subplot 3: Latency vs Speed Trade-off
ax3 = axes[1, 0]
for prov in sorted(df['Provider'].unique()):
    sub = df[df['Provider'] == prov]
    ax3.scatter(sub['Latency (sec)'], sub['Speed (tokens/sec)'],
                label=prov, alpha=0.75, s=60, edgecolors='none',
                color=PROVIDER_COLORS.get(prov, '#64748b'))
speed_pareto = df[df['Is_Speed_Pareto']].sort_values('Latency (sec)')
ax3.step(speed_pareto['Latency (sec)'], speed_pareto['Speed (tokens/sec)'],
         where='post', color='#b91c1c', linewidth=2, linestyle='--', label='Speed-Latency Pareto Frontier')
ax3.set_title('C. Latency vs Generation Speed Trade-off', fontweight='bold')
ax3.set_xlabel('Latency (seconds, lower is better)')
ax3.set_ylabel('Speed (tokens/sec, higher is better)')
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), borderaxespad=0.)

# Subplot 4: Average Price by Provider with Error Bars
ax4 = axes[1, 1]
prov_means = df.groupby('Provider')['Price / Million Tokens'].mean().sort_values()
prov_stds = df.groupby('Provider')['Price / Million Tokens'].std().loc[prov_means.index]
bar_colors = [PROVIDER_COLORS.get(p, '#3b82f6') for p in prov_means.index]
bars = ax4.barh(prov_means.index, prov_means.values, xerr=prov_stds.values, color=bar_colors, alpha=0.85, capsize=4, edgecolor='black')
ax4.set_title('D. Average Price / Million Tokens by Provider (±1 Std Dev)', fontweight='bold')
ax4.set_xlabel('Price / Million Tokens ($ USD)')
for bar, val in zip(bars, prov_means.values):
    ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'${val:.2f}', va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 0.88, 0.96])
fig1_path = os.path.join(OUTPUT_DIR, '01_matplotlib_executive_dashboard.png')
plt.savefig(fig1_path)
plt.close(fig1)
print(f"[+] Saved: {fig1_path}")

# ------------------------------------------------------------------------------
# Figure 2: Matplotlib Cost-Efficiency Pareto Frontier
# ------------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(13, 8))
for prov in sorted(df['Provider'].unique()):
    sub = df[df['Provider'] == prov]
    ax.scatter(sub['Price / Million Tokens'], sub['Benchmark (Chatbot Arena)'],
               color=PROVIDER_COLORS.get(prov, '#64748b'), label=prov,
               alpha=0.6, s=70, edgecolor='white', linewidth=0.5)

# Plot Pareto frontier line
pareto_sorted = df[df['Is_Cost_Pareto']].sort_values('Price / Million Tokens')
ax.plot(pareto_sorted['Price / Million Tokens'], pareto_sorted['Benchmark (Chatbot Arena)'],
        color='#dc2626', linewidth=2.5, marker='o', markersize=8, markerfacecolor='#fee2e2',
        markeredgecolor='#dc2626', markeredgewidth=2, label='State-of-the-Art Pareto Frontier')

# Annotate Pareto Frontier models
for _, row in pareto_sorted.iterrows():
    p_val = row['Price / Million Tokens']
    a_val = row['Benchmark (Chatbot Arena)']
    ax.annotate(f"{row['Model']} (${p_val:.2f}, {int(a_val)} Elo)",
                xy=(p_val, a_val),
                xytext=(p_val + 0.8, a_val - 18),
                fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#991b1b', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef2f2', edgecolor='#f87171', alpha=0.9))

ax.set_title('LLM Frontier Analysis: Quality (Chatbot Arena Elo) vs Cost ($/Million Tokens)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Price / Million Tokens ($ USD, lower is better)', fontsize=12)
ax.set_ylabel('Chatbot Arena Elo Score (higher is better)', fontsize=12)
ax.fill_between(pareto_sorted['Price / Million Tokens'], pareto_sorted['Benchmark (Chatbot Arena)'],
                df['Benchmark (Chatbot Arena)'].min() - 20, color='#fee2e2', alpha=0.2, label='Dominated Region')
ax.set_ylim(df['Benchmark (Chatbot Arena)'].min() - 30, df['Benchmark (Chatbot Arena)'].max() + 30)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
fig2_path = os.path.join(OUTPUT_DIR, '02_matplotlib_pareto_frontier.png')
plt.savefig(fig2_path)
plt.close(fig2)
print(f"[+] Saved: {fig2_path}")

# ------------------------------------------------------------------------------
# Figure 3: Compute Power vs Energy Efficiency (Green AI)
# ------------------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=df, x='Compute Power', y='Energy Efficiency',
    scatter_kws={'alpha': 0.6, 'color': '#0d9488', 's': 50},
    line_kws={'color': '#0f766e', 'linewidth': 2},
    ax=ax
)
ax.set_title('Compute Power vs Energy Efficiency Correlation & Regression', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Compute Power (FLOPs Index)', fontsize=11)
ax.set_ylabel('Energy Efficiency Rating (higher is greener)', fontsize=11)
ax.axvline(df['Compute Power'].median(), color='gray', linestyle=':', alpha=0.7)
ax.axhline(df['Energy Efficiency'].median(), color='gray', linestyle=':', alpha=0.7)
ax.text(df['Compute Power'].max() * 0.72, df['Energy Efficiency'].max() * 0.92, 'Green AI Leaders',
        color='#065f46', fontweight='bold', bbox=dict(boxstyle='round', facecolor='#d1fae5', edgecolor='#34d399', alpha=0.85))
fig3_path = os.path.join(OUTPUT_DIR, '03_matplotlib_compute_vs_energy.png')
plt.savefig(fig3_path)
plt.close(fig3)
print(f"[+] Saved: {fig3_path}")


# ------------------------------------------------------------------------------
# 6. Seaborn Visualizations (Statistical & Distribution Plots)
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("                    GENERATING SEABORN VISUALIZATIONS")
print("=" * 85)

# ------------------------------------------------------------------------------
# Figure 4: Correlation Heatmap
# ------------------------------------------------------------------------------
corr_matrix = df[[
    'Context Window', 'Speed (tokens/sec)', 'Latency (sec)',
    'Benchmark (MMLU)', 'Benchmark (Chatbot Arena)', 'Open-Source',
    'Price / Million Tokens', 'Training Dataset Size', 'Compute Power',
    'Energy Efficiency', 'Quality Rating', 'Speed Rating', 'Price Rating',
    'Throughput_Latency_Ratio', 'Cost_Performance_Ratio', 'Composite_Score'
]].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig4, ax = plt.subplots(figsize=(14, 11))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8,
    annot=True, fmt='.2f', square=True, linewidths=0.6,
    cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation Coefficient'},
    ax=ax, annot_kws={'size': 8}
)
ax.set_title('LLM Attribute Correlation Heatmap (Pearson Correlation)', fontsize=16, fontweight='bold', pad=15)
fig4_path = os.path.join(OUTPUT_DIR, '04_seaborn_correlation_heatmap.png')
plt.savefig(fig4_path)
plt.close(fig4)
print(f"[+] Saved: {fig4_path}")

# ------------------------------------------------------------------------------
# Figure 5: Provider Benchmark Distributions (Box + Strip Plot)
# ------------------------------------------------------------------------------
fig5, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
order_arena = df.groupby('Provider')['Benchmark (Chatbot Arena)'].median().sort_values(ascending=False).index

sns.boxplot(
    data=df, x='Provider', y='Benchmark (Chatbot Arena)', hue='Provider', order=order_arena,
    palette=PROVIDER_COLORS, legend=False, ax=axes[0], boxprops=dict(alpha=0.6), showfliers=False
)
sns.stripplot(
    data=df, x='Provider', y='Benchmark (Chatbot Arena)', order=order_arena,
    color='black', alpha=0.6, size=5, jitter=0.25, ax=axes[0]
)
axes[0].set_title('Chatbot Arena Elo Distribution by Provider', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Provider', fontweight='bold')
axes[0].set_ylabel('Chatbot Arena Elo Score', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

order_mmlu = df.groupby('Provider')['Benchmark (MMLU)'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df, x='Provider', y='Benchmark (MMLU)', hue='Provider', order=order_mmlu,
    palette=PROVIDER_COLORS, legend=False, ax=axes[1], boxprops=dict(alpha=0.6), showfliers=False
)
sns.stripplot(
    data=df, x='Provider', y='Benchmark (MMLU)', order=order_mmlu,
    color='black', alpha=0.6, size=5, jitter=0.25, ax=axes[1]
)
axes[1].set_title('MMLU Benchmark Score Distribution by Provider', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Provider', fontweight='bold')
axes[1].set_ylabel('MMLU Benchmark Score (%)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, '05_seaborn_benchmarks_by_provider.png')
plt.savefig(fig5_path)
plt.close(fig5)
print(f"[+] Saved: {fig5_path}")

# ------------------------------------------------------------------------------
# Figure 6: Open-Source vs Proprietary Architecture Comparison (Violin Plots)
# ------------------------------------------------------------------------------
fig6, axes = plt.subplots(2, 2, figsize=(14, 10))
fig6.suptitle('Open-Source vs Proprietary LLM Architecture Comparison', fontsize=16, fontweight='bold', y=0.99)
palette_os = {'Open Source': '#10b981', 'Proprietary': '#6366f1'}

sns.violinplot(data=df, x='Open_Source_Label', y='Price / Million Tokens', hue='Open_Source_Label', palette=palette_os, legend=False, inner='quartile', ax=axes[0, 0])
axes[0, 0].set_title('A. Price / Million Tokens ($)', fontweight='bold')
axes[0, 0].set_xlabel('')

sns.violinplot(data=df, x='Open_Source_Label', y='Speed (tokens/sec)', hue='Open_Source_Label', palette=palette_os, legend=False, inner='quartile', ax=axes[0, 1])
axes[0, 1].set_title('B. Generation Speed (tokens/sec)', fontweight='bold')
axes[0, 1].set_xlabel('')

sns.violinplot(data=df, x='Open_Source_Label', y='Latency (sec)', hue='Open_Source_Label', palette=palette_os, legend=False, inner='quartile', ax=axes[1, 0])
axes[1, 0].set_title('C. Response Latency (seconds)', fontweight='bold')
axes[1, 0].set_xlabel('')

sns.violinplot(data=df, x='Open_Source_Label', y='Benchmark (Chatbot Arena)', hue='Open_Source_Label', palette=palette_os, legend=False, inner='quartile', ax=axes[1, 1])
axes[1, 1].set_title('D. Chatbot Arena Elo Score', fontweight='bold')
axes[1, 1].set_xlabel('')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig6_path = os.path.join(OUTPUT_DIR, '06_seaborn_open_vs_proprietary.png')
plt.savefig(fig6_path)
plt.close(fig6)
print(f"[+] Saved: {fig6_path}")

# ------------------------------------------------------------------------------
# Figure 7: Speed vs Latency Joint Distribution with Marginal KDEs
# ------------------------------------------------------------------------------
g = sns.jointplot(
    data=df, x='Latency (sec)', y='Speed (tokens/sec)',
    hue='Open_Source_Label', palette=palette_os,
    kind='scatter', height=8, marginal_kws=dict(fill=True, common_norm=False),
    joint_kws=dict(alpha=0.7, s=60)
)
g.fig.suptitle('Generation Speed vs Latency Joint Distribution', fontsize=14, fontweight='bold', y=1.02)
g.set_axis_labels('Latency (sec)', 'Generation Speed (tokens/sec)')
fig7_path = os.path.join(OUTPUT_DIR, '07_seaborn_speed_vs_latency_jointplot.png')
g.savefig(fig7_path)
plt.close(g.fig)
print(f"[+] Saved: {fig7_path}")


# ------------------------------------------------------------------------------
# 7. Plotly Express Visualizations (Interactive Web Dashboards)
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("               GENERATING PLOTLY EXPRESS VISUALIZATIONS")
print("=" * 85)

# ------------------------------------------------------------------------------
# Plotly 1: 3D Interactive Frontier Explorer (Price vs Speed vs Chatbot Arena)
# ------------------------------------------------------------------------------
fig_3d = px.scatter_3d(
    df,
    x='Price / Million Tokens',
    y='Speed (tokens/sec)',
    z='Benchmark (Chatbot Arena)',
    color='Provider',
    size='Context Window',
    hover_name='Model',
    hover_data={
        'Provider': True,
        'Open_Source_Label': True,
        'Benchmark (MMLU)': True,
        'Latency (sec)': ':.2f',
        'Composite_Score': ':.2f',
        'Price / Million Tokens': ':.2f',
        'Speed (tokens/sec)': True,
        'Benchmark (Chatbot Arena)': True,
        'Context Window': ':,d'
    },
    color_discrete_map=PROVIDER_COLORS,
    opacity=0.85,
    title='<b>3D LLM Trade-Off Space: Cost vs Speed vs Reasoning Quality</b><br><sup>Size indicates Context Window; Color denotes Provider</sup>'
)
fig_3d.update_layout(
    scene=dict(
        xaxis_title='Price / Million Tokens ($)',
        yaxis_title='Speed (tokens/sec)',
        zaxis_title='Chatbot Arena Elo'
    ),
    template='plotly_white',
    margin=dict(l=0, r=0, b=0, t=50)
)
p1_path = os.path.join(OUTPUT_DIR, '08_plotly_3d_arena_speed_price.html')
fig_3d.write_html(p1_path)
print(f"[+] Saved interactive chart: {p1_path}")

# ------------------------------------------------------------------------------
# Plotly 2: Interactive Dynamic Cost vs Performance Bubble Chart
# ------------------------------------------------------------------------------
fig_bubble = px.scatter(
    df,
    x='Price / Million Tokens',
    y='Benchmark (Chatbot Arena)',
    size='Speed (tokens/sec)',
    color='Provider',
    facet_col='Open_Source_Label',
    hover_name='Model',
    hover_data={
        'Benchmark (MMLU)': True,
        'Latency (sec)': ':.2f',
        'Context Window': ':,d',
        'Energy Efficiency': ':.2f',
        'Composite_Score': ':.2f'
    },
    color_discrete_map=PROVIDER_COLORS,
    title='<b>Cost vs Performance Frontier: Chatbot Arena Elo vs Price / M Tokens</b><br><sup>Bubble size represents Generation Speed; Faceted by Open-Source vs Proprietary</sup>'
)
fig_bubble.update_layout(
    template='plotly_white',
    xaxis_title='Price / Million Tokens ($ USD)',
    yaxis_title='Chatbot Arena Elo Score',
    hovermode='closest'
)
p2_path = os.path.join(OUTPUT_DIR, '09_plotly_cost_vs_performance_bubble.html')
fig_bubble.write_html(p2_path)
print(f"[+] Saved interactive chart: {p2_path}")

# ------------------------------------------------------------------------------
# Plotly 3: Sunburst Chart of Provider & Model Hierarchy
# ------------------------------------------------------------------------------
fig_sunburst = px.sunburst(
    df,
    path=['Provider', 'Open_Source_Label', 'Model'],
    values='Context Window',
    color='Benchmark (Chatbot Arena)',
    color_continuous_scale='Blues',
    hover_data=['Benchmark (MMLU)', 'Speed (tokens/sec)', 'Price / Million Tokens'],
    title='<b>LLM Ecosystem Hierarchy: Provider -> Architecture -> Model</b><br><sup>Segment size = Context Window; Color = Chatbot Arena Elo Score</sup>'
)
fig_sunburst.update_layout(template='plotly_white')
p3_path = os.path.join(OUTPUT_DIR, '10_plotly_provider_hierarchy_sunburst.html')
fig_sunburst.write_html(p3_path)
print(f"[+] Saved interactive chart: {p3_path}")

# ------------------------------------------------------------------------------
# Plotly 4: Parallel Coordinates Chart (Multi-Dimensional Model Filter)
# ------------------------------------------------------------------------------
fig_parallel = px.parallel_coordinates(
    df,
    dimensions=[
        'Benchmark (Chatbot Arena)',
        'Benchmark (MMLU)',
        'Speed (tokens/sec)',
        'Latency (sec)',
        'Price / Million Tokens',
        'Energy Efficiency',
        'Compute Power'
    ],
    color='Composite_Score',
    color_continuous_scale=px.colors.diverging.Tealrose,
    title='<b>Multi-Dimensional LLM Trade-Off Explorer (Parallel Coordinates)</b><br><sup>Drag vertical axis intervals to filter models based on deployment SLAs</sup>'
)
fig_parallel.update_layout(template='plotly_white')
p4_path = os.path.join(OUTPUT_DIR, '11_plotly_multidimensional_parallel_coords.html')
fig_parallel.write_html(p4_path)
print(f"[+] Saved interactive chart: {p4_path}")

# ------------------------------------------------------------------------------
# Plotly 5: Interactive Provider Comparison Matrix Bar Chart
# ------------------------------------------------------------------------------
prov_agg_melted = provider_summary.reset_index().melt(
    id_vars=['Provider'],
    value_vars=['Arena_Mean', 'MMLU_Mean', 'Speed_Mean', 'Price_Mean'],
    var_name='Metric', value_name='Score'
)
metric_labels = {
    'Arena_Mean': 'Chatbot Arena (Elo / 10)',
    'MMLU_Mean': 'MMLU Score (%)',
    'Speed_Mean': 'Speed (tokens/sec)',
    'Price_Mean': 'Price ($ / M Tokens)'
}
prov_agg_melted['Normalized_Score'] = prov_agg_melted.apply(
    lambda r: r['Score'] / 10.0 if r['Metric'] == 'Arena_Mean' else r['Score'], axis=1
)
prov_agg_melted['Metric_Display'] = prov_agg_melted['Metric'].map(metric_labels)

fig_bars = px.bar(
    prov_agg_melted,
    x='Provider',
    y='Normalized_Score',
    color='Metric_Display',
    barmode='group',
    text='Score',
    title='<b>Provider Performance Metrics Comparison Matrix</b><br><sup>Compare Chatbot Arena, MMLU, Generation Speed, and Token Pricing Across All 8 Providers</sup>',
    color_discrete_sequence=['#2563eb', '#8b5cf6', '#10b981', '#f59e0b']
)
fig_bars.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig_bars.update_layout(
    template='plotly_white',
    xaxis_title='LLM Provider',
    yaxis_title='Normalized Metric Value',
    legend_title='Metric'
)
p5_path = os.path.join(OUTPUT_DIR, '12_plotly_provider_comparison_bars.html')
fig_bars.write_html(p5_path)
print(f"[+] Saved interactive chart: {p5_path}")


# ------------------------------------------------------------------------------
# 8. Execution Verification & Final Index
# ------------------------------------------------------------------------------
print("\n" + "=" * 85)
print("                     PIPELINE EXECUTION COMPLETE!")
print("=" * 85)
print("Generated Outputs:")
print("  [Static High-Resolution Charts - Matplotlib & Seaborn]")
print(f"   1. {fig1_path}")
print(f"   2. {fig2_path}")
print(f"   3. {fig3_path}")
print(f"   4. {fig4_path}")
print(f"   5. {fig5_path}")
print(f"   6. {fig6_path}")
print(f"   7. {fig7_path}")
print("  [Interactive HTML Dashboards - Plotly Express]")
print(f"   8. {p1_path}")
print(f"   9. {p2_path}")
print(f"  10. {p3_path}")
print(f"  11. {p4_path}")
print(f"  12. {p5_path}")
print("  [Excel Multi-Sheet Summary Workbook]")
print(f"  -> {summary_excel_path}")
print("=" * 85)
