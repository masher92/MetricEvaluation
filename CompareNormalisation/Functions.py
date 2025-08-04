import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from adjustText import adjust_text
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, LogFormatter, FuncFormatter
from matplotlib.patches import Patch

import MetricMapping
    
# resolution_index_norm = {'5m':0, '10m': 1, '30m': 2, '60m': 3}  
resolution_index_res = {'10m': 0, '30m': 1, '60m': 2}
    
def spaced_colors_from_cmap(cmap_name, values=None, add_grey=False):
    if isinstance(values, int):  # if user passed number of colors
        values = np.linspace(0.2, 0.8, values)  # avoid extremes of cmap
    elif values is None:
        values = [0.8]  # default single value

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(v)[:3] for v in values]  # RGB only
    if add_grey:
        grey = (0.6, 0.6, 0.6)
        colors = [grey] + colors
    return colors


type_color_map_1 = {
    'Asymmetry': spaced_colors_from_cmap("Blues", 3, add_grey=True),
    'Peakiness': spaced_colors_from_cmap("Reds", 3, add_grey=True),
    'Concentration': spaced_colors_from_cmap("Greens", 3, add_grey=True),
    'Intermittency': spaced_colors_from_cmap("Purples", 3, add_grey=True),
    'Categorical': spaced_colors_from_cmap("Oranges", 3, add_grey=True),}

type_color_map_2 = {
    'Asymmetry': sns.color_palette("Blues", 3),
    'Peakiness': sns.color_palette("Reds", 3),
    'Concentration': sns.color_palette("Greens", 3),
    'Intermittency': sns.color_palette("Purples", 3),
    'categorical': sns.color_palette("Oranges", 3)}


type_color_map_3 = {
    'Asymmetry': spaced_colors_from_cmap("Blues",  add_grey=True),
    'Peakiness': spaced_colors_from_cmap("Reds", add_grey=True),
    'Concentration': spaced_colors_from_cmap("Greens",  add_grey=True),
    'Intermittency': spaced_colors_from_cmap("Purples",  add_grey=True),
    'Categorical': spaced_colors_from_cmap("Oranges",  add_grey=True),}

type_mapping = MetricMapping.type_mapping
name_mapping = MetricMapping.name_mapping

def scatter_without_labels(ax, data, this_metric, type_color_map, resolution_index_res):
    for _, row in data.iterrows():
        metric_type = row["type2"]
        resolution = row["resolution"]
        color = type_color_map[metric_type][resolution_index_res[resolution]]
        ax.scatter(row["rank_corr"], row["val_diff"], color=color, edgecolor='black',
                   marker='o', s=350, alpha=1)
    ax.set_title(this_metric, fontsize=25, fontstyle="italic")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_xlabel("Spearman’s ρ", fontsize=15)
    ax.set_ylabel("MAD from 5m", fontsize=15)
    
    # Highlight box
    x_min, x_max = 0.9, 1.001
    y_min, y_max = 0, 0.2
    
    ax.tick_params(axis='both', which='major', labelsize=15)


# Scatter function
def scatter_without_labels_cat(ax, data, this_metric, type_color_map, resolution_index):
    for _, row in data.iterrows():
        metric_type = row["type"]
        resolution = row["resolution"]
        color = type_color_map[metric_type][resolution_index[resolution]]
        ax.scatter(row["rank_corr"], row["val_diff"], color=color, edgecolor='black',
                   marker='o', s=350, alpha=1)
    ax.set_title(this_metric, fontsize=25,fontstyle="italic")
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_xlabel("Kendall’s τ", fontsize=15)
    ax.set_ylabel("% Different from 5m", fontsize=15)
    ax.set_ylim(0,70)
    x_min, x_max = 0.8, 1.001  
    y_min, y_max = 0, 10  
    ax.tick_params(axis='both', which='major', labelsize=15)


def split_metric_resolution(col_name):
    """
    Splits column names like '3rd_w_Peak_5m' into ('3rd_w_Peak', '5m').
    Assumes resolution is always the last underscore-suffix (e.g., '_5m').
    """
    parts = col_name.rsplit('_', 1)  # Split only on the last underscore
    if len(parts) == 2 and parts[1] in ['5m', '30m']:
        return parts[0], parts[1]
    else:
        return col_name, None  # No valid resolution suffix found

def compute_metric_sensitivity_by_resolution(df, continuous_metrics, categorical_metrics, resolutions=["10m", "30m", "60m"]):
    rows = []

    for res in resolutions:
        for metric in continuous_metrics + categorical_metrics:
            ref_col = f"{metric}_5m"
            comp_col = f"{metric}_{res}"
            if ref_col not in df.columns or comp_col not in df.columns:
                continue

            x_vals = df[ref_col]
            y_vals = df[comp_col]
            valid = x_vals.notna() & y_vals.notna()
            x = x_vals[valid]
            y = y_vals[valid]

            if len(x) < 2:
                continue

            is_continuous = metric in continuous_metrics
            if is_continuous:
                rank_corr, _ = spearmanr(x, y)
#                 val_diff = np.mean(np.abs(y - x))  # MAD
                val_diff = np.median(np.abs((y - x) / np.where(x == 0, np.nan, x)) * 100)
                val_diff = 100 * np.mean(np.abs(y - x) / ((np.abs(x) + np.abs(y)) / 2))
            else:
                rank_corr, _ = kendalltau(x, y)
                observed_diff = np.mean(x != y) * 100  # raw % different
                observed_diff = observed_diff*100
                # Option A: Normalize based on number of classes in 5-min data
                n_classes = x.nunique()
                if n_classes > 1:
                    max_diff = (1 - 1 / n_classes) * 100  # convert to percent
                    val_diff = observed_diff / max_diff  # normalized disagreement
                else:
                    val_diff = 0  # No disagreement possible if only one class

            spread = gini(y)

            rows.append({
                "metric": metric,
                "resolution": res,
                "type": "continuous" if is_continuous else "categorical",
                "rank_corr": rank_corr,
                "val_diff": val_diff,
                "gini": spread
            })

    return pd.DataFrame(rows)    
    
    
def compute_metric_sensitivity_bynormalisation(df, continuous_metrics, categorical_metrics, resolutions=["DMC_10"]):
    rows = []

    for res in resolutions:
        for metric in continuous_metrics + categorical_metrics:
            ref_col = f"{metric}"
            comp_col = f"{metric}_{res}"
            if ref_col not in df.columns or comp_col not in df.columns:
                continue

            x_vals = df[ref_col]
            y_vals = df[comp_col]
            valid = x_vals.notna() & y_vals.notna()
            x = x_vals[valid]
            y = y_vals[valid]

            if len(x) < 2:
                continue

            is_continuous = metric in continuous_metrics
            if is_continuous:
                rank_corr, _ = spearmanr(x, y)
#                 val_diff = np.mean(np.abs(y - x))  # MAD
                # val_diff = np.median(np.abs((y - x) / np.where(x == 0, np.nan, x)) * 100)
                val_diff = 100 * np.mean(np.abs(y - x) / ((np.abs(x) + np.abs(y)) / 2))
            else:
                rank_corr, _ = kendalltau(x, y)
                observed_diff = np.mean(x != y) * 100  # raw % different
                observed_diff = observed_diff*100
                # Option A: Normalize based on number of classes in 5-min data
                n_classes = x.nunique()
                if n_classes > 1:
                    max_diff = (1 - 1 / n_classes) * 100  # convert to percent
                    val_diff = observed_diff / max_diff  # normalized disagreement
                else:
                    val_diff = 0  # No disagreement possible if only one class

            spread = gini(y)

            rows.append({
                "metric": metric,
                "resolution": res,
                "type": "continuous" if is_continuous else "categorical",
                "rank_corr": rank_corr,
                "val_diff": val_diff,
                "gini": spread
            })

    return pd.DataFrame(rows)

def smart_log_tick_format(x, pos):
    if x == 0:
        return "0"  # Not expected on log axis
    elif x >= 1:
        return f"{x:g}"  # Plain formatting
    else:
        decimals = int(np.ceil(-np.log10(x)))
        if decimals > 3:
            return f"{x:.0e}"  # Scientific notation
        else:
            return f"{x:.{decimals}f}"

def gini(array):
    array = np.sort(np.array(array))
    n = len(array)
    if n == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)) if np.sum(array) != 0 else 0

def plot_histograms(ax, transformed_minmax_scaled, metric, metric_type_df, log_scale_metrics, type_color_map, resolutions):
    label_resolutions=['5 minute', "10 minute", "30 minute", "60 minute"]
    for num, res in enumerate(resolutions):
        col_name = f"{metric}{res}"
        this_type = metric_type_df[metric_type_df['metric'] == metric]['type2'].iloc[0]
        color_map = type_color_map[this_type]
        
        if col_name in transformed_minmax_scaled.columns:
            values = transformed_minmax_scaled[col_name].dropna()

            # Use log bins for specified metrics
            if metric in log_scale_metrics:
                values = values[values > 0]  # Avoid log of 0 or negative
                if len(values) > 0:
                    bins = 50 # np.logspace(np.log10(values.min()), np.log10(values.max()), 10)
                    ax.set_xscale("log")
                    ax.xaxis.set_major_formatter(FuncFormatter(smart_log_tick_format))
                else:
                    continue  # Skip empty or non-positive
            else:
                bins = 20
            sns.histplot(
                values,
                bins=bins,
                kde=False,
                ax=ax,
                color=color_map[resolutions.index(res)],
                label=label_resolutions[num],
                element='step',
                stat='density',
                fill=True,
                alpha=0.8)

    title = name_mapping[metric]    
    ax.set_title(title, fontsize=27,fontstyle="italic")
    ax.set_xlabel('')
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True)

def plot_grouped_categorical(ax, metric, df, type_color_map, resolutions):
    # 1. build normalized counts per category per resolution
    vc = {res: df[f"{metric}{res}"].value_counts(normalize=True)
        for res in resolutions}
    vc_df = pd.DataFrame(vc).fillna(0)        # index = category values
    vc_df = vc_df.sort_index()               # ensure logical order of categories
    categories = vc_df.index.tolist()

    # Adjust x-axis labels if numeric categories (0-indexed integers)
    categories = range(0,len(categories))
    x_labels = categories  # use original labels if not numeric
    x_labels = [str(int(cat) + 1) for cat in categories]
        
    # 2. bar positions
    x = np.arange(len(categories))
    n = len(resolutions)
    total_width = 0.8
    width = total_width / n

    # 3. fetch colors for this metric’s type
    palette = type_color_map['Categorical']

    # 4. draw each resolution’s bars (convert to percentage)
    for i, res in enumerate(resolutions):
        heights = vc_df[res].values * 100  # convert to percent
        offset = (i - (n-1)/2) * width
        ax.bar( x + offset, heights, width=width, label=res,
            color=palette[i], edgecolor='white', alpha=1)

    # 5. formatting
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of events", fontsize=20)
    title = name_mapping[metric]    
    ax.set_title(title, fontsize=30,fontstyle="italic")
    ax.grid(True)