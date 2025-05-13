from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

def plot_mass_curve_dual(rainfall, df, suffix, save_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    def get_non_overlapping_y(candidate_y, used_ys, spacing, line_y):
        for y in used_ys:
            if abs(candidate_y - y) < spacing:
                return get_non_overlapping_y(candidate_y + spacing, used_ys, spacing, line_y)
        return candidate_y

    if suffix in ['_norm', '_dblnorm']:
        cumulative_rainfall = rainfall.iloc[:, 0].to_numpy()
        incremental_rainfall = np.diff(cumulative_rainfall, prepend=0)
        times = np.linspace(0, 1, len(rainfall))
    else:
        times = rainfall.index
        incremental_rainfall = rainfall.iloc[:, 0].to_numpy()
        cumulative_rainfall = np.cumsum(incremental_rainfall)

    total = cumulative_rainfall[-1]
    if total == 0:
        print("Total rainfall is zero; nothing to plot.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    axs[0].plot(times, cumulative_rainfall, label='Cumulative Rainfall', color='black', linewidth=3)

    idxs = ['m4', 'm5']
    time_fractions = [0.30,  0.50]
    spacing = total * 0.07
    event_duration = times[-1] - times[0]
    steps = len(cumulative_rainfall)

    used_text_xs = []
    for num, fraction in enumerate(time_fractions):
        time_at_fraction = fraction * event_duration + times[0]
        idx = (np.abs(times - time_at_fraction)).argmin()
        time_at_fraction = times[idx]
        rainfall_at_time = cumulative_rainfall[idx]

        axs[0].fill_between(times, 0, cumulative_rainfall, where=(times <= time_at_fraction), 
                            color='lightblue', alpha=0.5, label=f'{int(fraction*100)}% of Event')
        axs[0].plot(time_at_fraction, rainfall_at_time, 'ro')

        fraction_rainfall = rainfall_at_time / total
        text_position_y = rainfall_at_time * 0.55
        text_x = time_at_fraction - 0.1 if not isinstance(time_at_fraction, (pd.Timestamp, pd.DatetimeIndex, np.datetime64)) else pd.to_datetime(time_at_fraction) - pd.Timedelta(minutes=5)

        # Check for x overlaps
        while text_x in used_text_xs:
            text_x += 0.05 if not isinstance(text_x, pd.Timestamp) else pd.Timedelta(minutes=2)
        used_text_xs.append(text_x)

        axs[0].text(text_x, text_position_y, f'{idxs[num]}: {fraction_rainfall*100:.1f}%\n in {fraction} of event', 
                    color='blue', ha='center')

    used_ys = []
    for dx_value, color in zip([25, 50, 75], ['darkorange', 'purple', 'teal']):
        dx_frac = df[f'T{dx_value}{suffix}'][0] / 100  # Fraction (e.g., 0.5 for 50%)
        dx_y = dx_value/100
        dx_y = dx_y * total  # Actual rainfall value for this Dx        
        cumulative = cumulative_rainfall  # Not normalized — use raw cumulative rainfall

        # Interpolate to find x (time) where cumulative crosses dx_y
        for i in range(1, len(cumulative)):
            if cumulative[i-1] <= dx_y <= cumulative[i]:
                x0, x1 = times[i-1], times[i]
                y0, y1 = cumulative[i-1], cumulative[i]
                dx_time = x0 + (dx_y - y0) * (x1 - x0) / (y1 - y0)
                break
        else:
            dx_time = times[-1]  # Fallback

        # Horizontal dashed line from y-axis to intersection
        axs[0].hlines(dx_y, times[0], dx_time, color=color, linestyle='dashed')

        # Vertical dashed arrow from intersection to x-axis
        axs[0].annotate("", xy=(dx_time, 0), xytext=(dx_time, dx_y),
                        arrowprops=dict(arrowstyle="->", linestyle='dashed', lw=1.2, color=color))

        # Avoid overlapping annotation text
        annot_y = get_non_overlapping_y(dx_y + total * 0.05, used_ys, spacing, line_y=dx_y)
        used_ys.append(annot_y)

        # Add label with arrow
        axs[0].annotate(f"D{dx_value} = {dx_frac*100:.1f}%", xy=(dx_time, dx_y),
                        xytext=(dx_time, annot_y),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
                        fontsize=14, color=color)


    fifth_w_peak = df[f'5th_w_peak{suffix}']
    fifth_size = steps // 5
    fifth_edges = [i * fifth_size for i in range(5)] + [steps - 1]
    fifth_w_peak_i = int(fifth_w_peak) + 1
    start_idx, end_idx = fifth_edges[fifth_w_peak_i - 1], fifth_edges[fifth_w_peak_i]

    y_values = cumulative_rainfall
    axs[0].fill_between(times[start_idx:end_idx + 1], 0, y_values[start_idx:end_idx + 1], color='lightgreen', alpha=0.8)
    for edge in fifth_edges[fifth_w_peak_i - 1:fifth_w_peak_i + 1]:
        if edge < steps:
            axs[0].plot([times[edge], times[edge]], [0, y_values[edge]], color='grey', linestyle='solid', linewidth=2)

    mid_idx = (start_idx + end_idx) // 2
    mid_time = times[mid_idx]
    mid_y = y_values[mid_idx] * 0.5
    axs[0].text(mid_time, mid_y, f"5th with max is {fifth_w_peak_i}", ha='center', va='center', fontsize=12, color='black')

    peak_idx = np.argmax(incremental_rainfall)
    axs[0].annotate('', xy=(times[peak_idx], total * 0.98), xytext=(times[0], total * 0.98),
                    arrowprops=dict(arrowstyle='<->', lw=2, color='darkgreen'))
    time_to_peak = times[peak_idx] - times[0]
    time_to_peak_str = f"{(time_to_peak*100):.1f}%" if suffix != '' else f"{time_to_peak}"
    axs[0].text(times[peak_idx // 2], total * 0.99, f"Time to Peak = {time_to_peak_str}", color='darkgreen', ha='center', fontsize=10)

    axs[0].set_ylabel("Rainfall (mm/hr)" if suffix == '' else "Cumulative Fraction")
    axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # === Centre of Gravity (Centre of Mass) ===
    rcg = df[f'centre_gravity{suffix}']
    if suffix in ['_dblnorm', '_norm']:
        x_pos = rcg[0]
    else:
        x_idx = int(round(rcg * (len(times) - 1)))
        x_pos = times[x_idx]

    idx = (np.abs(times - x_pos)).argmin()
    y_value = cumulative_rainfall[idx]

    # Calculate vertical distance from D50 (or any close metric)
    d50_frac = df[f'T50{suffix}'][0] / 100
    d50_idx = int(round(d50_frac * (steps - 1)))
    d50_y = cumulative_rainfall[d50_idx]

    # Check if overlap is likely (adjust threshold as needed)
    min_y_gap = total * 0.05  # Minimum vertical space
    if abs(y_value - d50_y) < min_y_gap:
        y_annot = y_value + min_y_gap  # Push upwards
    else:
        y_annot = y_value + total * 0.1

    # Handle x-position offset if needed
    if isinstance(x_pos, (pd.Timestamp, pd.DatetimeIndex, np.datetime64)):
        x_text = pd.to_datetime(x_pos) + pd.Timedelta(minutes=5)
    else:
        x_text = x_pos + 0.05

    axs[0].plot([x_pos, x_pos], [0, y_value], color='magenta', linestyle='dashed')
    axs[0].plot(x_pos, y_value, 'o', color='magenta', markersize=8)

#     axs[0].annotate(f"Centre of Mass {rcg[0]*100:.1f}%",
#                     xy=(x_pos, y_value),
#                     xytext=(x_text, y_annot),
#                     arrowprops=dict(arrowstyle="->", lw=1.5, color='magenta'),
#                     fontsize=14, color='magenta', ha='center')


    before_color = 'lightcoral'
    after_color = 'lightblue'
    if suffix in ['_norm', '_dblnorm']:
        axs[1].axvspan(0, times[peak_idx], color=before_color, alpha=0.2, label='Before Peak')
        axs[1].axvspan(times[peak_idx], 1, color=after_color, alpha=0.2, label='After Peak')
    else:
        axs[1].axvspan(times[0], times[peak_idx], color=before_color, alpha=0.2, label='Before Peak')
        axs[1].axvspan(times[peak_idx], times[-1], color=after_color, alpha=0.2, label='After Peak')

    rain_before_peak = np.sum(incremental_rainfall[:peak_idx + 1])
    rain_after_peak = np.sum(incremental_rainfall[peak_idx + 1:])
    m1 = rain_before_peak / rain_after_peak if rain_after_peak != 0 else np.inf
    m2 = np.max(incremental_rainfall) / np.sum(incremental_rainfall)

    summary_text = (f"Rainfall before peak:\n  {rain_before_peak:.2f} mm\n\n"
                    f"Rainfall after peak:\n  {rain_after_peak:.2f} mm\n\n"
                    f"m1 (before/after):\n  {m1:.2f} \n\n"
                    f"m2 (rain in peak/whole event):\n  {m2:.2f}")

    axs[0].text(1.02, 0.8, summary_text, transform=axs[0].transAxes,
                fontsize=12, color='black', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='grey', alpha=0.9))

    axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

    if suffix in ['_norm', '_dblnorm']:
        bar_width = 1 / steps
        bar_x = np.linspace(bar_width / 2, 1 - bar_width / 2, steps)
        axs[1].bar(bar_x, incremental_rainfall, width=bar_width, align='center', color='steelblue', alpha=0.7, edgecolor='black')
        axs[1].set_xlim(0, 1)
        axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    else:
        bar_x = times
        if isinstance(times[0], pd.Timestamp):
            time_deltas = np.diff(times) / np.timedelta64(1, 's')
            avg_dt = np.mean(time_deltas)
            bar_width = pd.to_timedelta(avg_dt, unit='s')
        else:
            bar_width = (times[1] - times[0])

        axs[1].bar(bar_x, incremental_rainfall, width=bar_width,
                   align='center', color='steelblue', alpha=0.7, edgecolor='black')
        axs[1].set_xlim(times[0], times[-1])

    axs[1].set_ylabel("Rainfall\n(mm)" if suffix == '' else "Rainfall\n(Fraction)")
    axs[1].set_xlabel("Time" if suffix == '' else "Dimensionless Time")
    axs[1].set_ylim(bottom=0)
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=0, top=incremental_rainfall.max() * 1.1)

    plt.tight_layout()
    plt.savefig(save_path)

    
def get_non_overlapping_y(proposed_y, used_ys, spacing, line_y=None):
    """Ensure y-position for text doesn't overlap previously used ones."""
    while any(abs(proposed_y - y) < spacing for y in used_ys):
        proposed_y += spacing
    return proposed_y



def plot_mass_curve_dual_old(rainfall, df, suffix):

    if suffix in ['_norm', '_dblnorm']:
        cumulative_rainfall = rainfall.iloc[:, 0].to_numpy()
        incremental_rainfall = np.diff(cumulative_rainfall, prepend=0)
        times = np.linspace(0, 1, len(rainfall))  # Dimensionless time
    else:
        times = rainfall.index
        incremental_rainfall = rainfall.iloc[:, 0].to_numpy()
        cumulative_rainfall = np.cumsum(incremental_rainfall)
    
    steps = len(cumulative_rainfall)
    total = cumulative_rainfall[-1]
    
    # Mass indicators
    peak_idx = np.argmax(incremental_rainfall)
    cum_fraction = cumulative_rainfall / total  # still needed for some things

    if total == 0:
        print("Total rainfall is zero; nothing to plot.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

    # === Top Plot ===
    axs[0].plot(times, cumulative_rainfall, label='Cumulative Rainfall', color='black', linewidth=3)
    axs[0].axvline(times[peak_idx], color='darkgreen', linestyle='--')

    rcg = df['centre_gravity']
    if suffix in ['_dblnorm', '_norm']:
        x_pos = rcg
    else:
        x_idx = int(round(rcg * (len(times) - 1)))
        x_pos = times[x_idx]

    axs[0].axvline(x_pos, color='magenta', linestyle='dashed')
    axs[0].annotate(f"Centre of Mass {rcg*100:.1f}%",
                    xy=(x_pos, total * 0.5),
                    xytext=(x_pos, total * 0.7),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='magenta'),
                    fontsize=14, color='magenta')
    
    # D25, D50, D75
    used_ys = []
    spacing = total * 0.07  # tweak this value to increase spacing between labels

    for dx_value, color in zip([25, 50, 75], ['darkorange', 'purple', 'teal']):
        dx_frac = df[f'T{dx_value}{suffix}'] / 100
        dx_idx = int(round(dx_frac * (steps - 1)))
        dx_y = cumulative_rainfall[dx_idx]

        # Plot vertical and horizontal lines
        axs[0].annotate("",
                        xy=(times[dx_idx], 0),             # Arrowhead at the x-axis
                        xytext=(times[dx_idx], dx_y),      # Tail at the data point
                        arrowprops=dict(arrowstyle="->",   # Arrow style
                                        linestyle='dotted',
                                        lw=1.2,
                                        color=color))

        axs[0].hlines(dx_y, times[0], times[dx_idx], color=color, linestyle='dotted')

        # Find a y position that doesn’t overlap
        annot_y = get_non_overlapping_y(dx_y + total * 0.05, used_ys, spacing, line_y=dx_y)
        used_ys.append(annot_y)

        # Add annotation
        axs[0].annotate(f"D{dx_value} = {dx_frac*100:.1f}%",
                        xy=(times[dx_idx], dx_y),
                        xytext=(times[dx_idx], annot_y),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
                        fontsize=14, color=color)
        
    # m3, m4, m5 lines
    m3_idx = int(np.round(steps / 3)) - 1
    m4_idx = int(np.round(steps * 0.3)) - 1
    m5_idx = int(np.round(steps / 2)) - 1       
    
    idx_name = ['m3', 'm4', 'm5']
    idx_list = [m3_idx, m4_idx, m5_idx]
    cols = ['red', 'darkblue', 'cyan']        
    for num, idx in enumerate(idx_list):
        val = cumulative_rainfall[idx]

        # Plot vertical and horizontal indicators
        axs[0].plot([times[idx], times[idx]], [0, val], color=cols[num], linestyle='dashdot')
        axs[0].annotate("",
                        xy=(times[0], val),                    # Arrowhead at y-axis
                        xytext=(times[idx], val),              # Tail at the vertical line
                        arrowprops=dict(arrowstyle='-|>',      # Line with arrowhead
                                        linestyle='dashdot',
                                        lw=1.2,
                                        color=cols[num]))


        # Determine a y position that doesn't overlap
        annot_y = get_non_overlapping_y(val + total * 0.05, used_ys, spacing, line_y=val)
        used_ys.append(annot_y)
        
        if suffix == '':
            extra = 'mm/hr'
        else:
            extra = '%'

        # Annotate
        axs[0].annotate(f"{idx_name[num]} = {val:.2f}{extra}",
                        xy=(times[idx], val),
                        xytext=(times[idx], annot_y),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=cols[num]),
                        fontsize=14, color=cols[num])
        

    # Peak time arrow
    axs[0].annotate('', xy=(times[peak_idx], total * 0.98), xytext=(times[0], total * 0.98),
                    arrowprops=dict(arrowstyle='<->', lw=2, color='darkgreen'))
    time_to_peak = times[peak_idx] - times[0]
    print(time_to_peak)
    time_to_peak_str = f"{(time_to_peak*100):.1f}%" if suffix != '' else f"{time_to_peak}"
    axs[0].text(times[peak_idx // 2], total * 0.99,
                f"Time to Peak = {time_to_peak_str}", color='darkgreen', ha='center', fontsize=10)


    axs[0].set_ylabel("Rainfall (mm/hr)" if suffix == '' else "Cumulative Fraction")
    axs[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Fifth shading
    fifth_w_peak = df[f'5th_w_peak{suffix}']
    fifth_size = steps // 5
    fifth_edges = [i * fifth_size for i in range(5)]
    fifth_edges.append(steps - 1)  # Ensure the last edge reaches the final index

    i = int(fifth_w_peak)
    start_idx = fifth_edges[i - 1]
    end_idx = fifth_edges[i]

    # Determine the y-values for the shaded area
    y_values = cumulative_rainfall if suffix == '' else cum_fraction

    # Fill the shaded area
    axs[0].fill_between(
        times[start_idx:end_idx + 1],
        0,
        y_values[start_idx:end_idx + 1],
        color='lightgreen',
        alpha=0.3
    )

    # Draw vertical lines at the edges of the shaded area
    for edge in fifth_edges[i - 1:i + 1]:
        if edge < steps:
            axs[0].plot(
                [times[edge], times[edge]],
                [0, y_values[edge]],
                color='grey',
                linestyle='solid',
                linewidth=2
            )

    # Add text annotation inside the shaded area
    mid_idx = (start_idx + end_idx) // 2
    mid_time = times[mid_idx]
    mid_y = y_values[mid_idx] * 0.5  # Position text at 50% of the peak value

    axs[0].text(mid_time,mid_y,f"5th with max is {i}", ha='center', va='center', fontsize=12, color='black')
    
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=0)
    
    # === Bottom Plot: Incremental Rainfall ===
    # === Bottom Plot: Incremental Rainfall ===
    if suffix in ['_norm', '_dblnorm']:
        bar_width = 1 / steps
        bar_x = np.linspace(bar_width / 2, 1 - bar_width / 2, steps)  # center bars in [0,1]
        axs[1].bar(bar_x, incremental_rainfall, width=bar_width, align='center',
                   color='steelblue', alpha=0.7, edgecolor='black')
        axs[1].set_xlim(0, 1)
        axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    else:
        bar_x = times
        if isinstance(times[0], pd.Timestamp):
            time_deltas = np.diff(times) / np.timedelta64(1, 's')  # in seconds
            avg_dt = np.mean(time_deltas)
            bar_width = pd.to_timedelta(avg_dt, unit='s')
        else:
            bar_width = (times[1] - times[0])

        axs[1].bar(bar_x, incremental_rainfall, width=bar_width,
                   align='center', color='steelblue', alpha=0.7, edgecolor='black')
        axs[1].set_xlim(times[0], times[-1])

    axs[1].set_ylabel("Rainfall\n(mm)" if suffix == '' else "Rainfall\n(Fraction)")
    axs[1].set_xlabel("Time" if suffix == '' else "Dimensionless Time")
    axs[1].set_ylim(bottom=0)
    fig.savefig("metric_demo_DMC.png", dpi=300, facecolor='white', edgecolor='white',)

def get_non_overlapping_y(y_value, existing_ys, spacing, line_y=None):
    new_y = y_value
    while any(abs(new_y - y) < spacing for y in existing_ys) or (line_y is not None and abs(new_y - line_y) < spacing):
        new_y += spacing
    return new_y

def get_non_overlapping_position(x, y, used_positions, spacing, avoid_xs=None, avoid_ys=None):
    max_tries = 20
    tries = 0
    new_y = y

    while tries < max_tries:
        collision = (
            any(abs(new_y - used_y) < spacing for _, used_y in used_positions) or
            (avoid_ys and any(abs(new_y - yline) < spacing for yline in avoid_ys)) or
            (avoid_xs and any(abs(x - xline) < spacing for xline in avoid_xs))
        )

        if not collision:
            return new_y

        new_y += spacing
        tries += 1
    return new_y  # fallback if too many tries



def plot_sorted_metric_scatter_grid(
    all_events_df,
    comparison_df,
    suffix1='_norm',
    suffix2='_dblnorm',
    sort_by='pearson_r'
):
    """
    Plots a grid of density scatter plots (Raw vs DMC) for each metric, sorted by similarity.

    Args:
        all_events_df: DataFrame with metric values.
        comparison_df: Output from compare_metrics_from_df().
        suffix1: Suffix for raw metrics (e.g. '_norm').
        suffix2: Suffix for transformed metrics (e.g. '_dblnorm').
        sort_by: Metric to sort plots by, e.g. 'pearson_r'.
    """
    # Sort metrics by similarity score (descending)
    sorted_metrics = comparison_df.sort_values(by=sort_by, ascending=False)['metric'].tolist()
    n_metrics = len(sorted_metrics)

    ncols = 8
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, metric in enumerate(sorted_metrics):
        
        col1 = f'{metric}{suffix1}'
        col2 = f'{metric}{suffix2}'

        if col1 not in all_events_df.columns or col2 not in all_events_df.columns:
            continue

        x = all_events_df[col1]
        y = all_events_df[col2]

        # Drop NaNs
        mask = ~x.isna() & ~y.isna()
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        ax = axs[i]
        if metric in ['3rd_ARR', '3rd_com', '3rd_maxpct', '3rd_w_peak']:
            bins=3
        elif metric in ['4th_w_peak']:
            bins=4
        elif metric in ['5th_w_peak']:
            bins=5
        else:
            bins=30
        
        sns.histplot(x=x, y=y, bins=bins, pmax=0.9, ax=ax, cmap="viridis", cbar=False)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=1)
        
        if metric == 'frac_rain_in_high_intensity_zone':
            metric = '% rain HIZ',
        if metric == 'frac_rain_in_low_intensity_zone':
            metric = '% rain LIZ',
        if metric == 'frac_time_in_high_intensity_zone':
            metric= '% time HIZ',
        if metric == 'frac_time_in_low_intensity_zone':
            metric= '% time LIZ',                
        
        ax.set_title(f"{metric}", fontsize=15)
        ax.set_xlabel('Raw')
        ax.set_ylabel('DMC')

    # Hide unused axes
    for j in range(n_metrics, len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    fig.savefig("sorted_metric_comparison_grid.png", dpi=300, facecolor='white')
    plt.show()
    
    
def plot_sorted_metric_difference_histograms(
    all_events_df,
    comparison_df,
    suffix1='_norm',
    suffix2='_dblnorm',
    sort_by='pearson_r'
):
    """
    Plots a grid of histograms showing relative differences (|raw - dmc| / |raw|) for each metric.

    Args:
        all_events_df: DataFrame with metric values.
        comparison_df: Output from compare_metrics_from_df().
        suffix1: Suffix for raw metrics (e.g. '_norm').
        suffix2: Suffix for transformed metrics (e.g. '_dblnorm').
        sort_by: Metric to sort plots by, e.g. 'pearson_r'.
    """
    # Sort metrics by similarity score (descending)
    sorted_metrics = comparison_df.sort_values(by=sort_by, ascending=False)['metric'].tolist()
    n_metrics = len(sorted_metrics)

    ncols = 6
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, metric in enumerate(sorted_metrics):
        col1 = f'{metric}{suffix1}'
        col2 = f'{metric}{suffix2}'

        if col1 not in all_events_df.columns or col2 not in all_events_df.columns:
            continue

        x = all_events_df[col1]
        y = all_events_df[col2]

        # Drop NaNs
        mask = ~x.isna() & ~y.isna()
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        # Compute relative difference
        rel_diff = np.abs(x - y) / (np.abs(x) + 1e-8)

        ax = axs[i]
        ax.hist(rel_diff, bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'{metric}', fontsize=10)
        ax.set_xlabel('|raw - dmc| / |raw|')
        ax.set_ylabel('Count')

    # Hide unused axes
    for j in range(n_metrics, len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    fig.savefig("sorted_metric_difference_histograms.png", dpi=300)
    plt.show()    

def plot_metrics_grid_with_histograms(all_events_df, comparison_df, suffix1='_norm', suffix2='_dblnorm', sort_by='pearson_r'):
    """
    Plots a grid of 2 plots per metric (scatter + rel. diff. histogram), ordered by similarity.
    
    Args:
        all_events_df: DataFrame containing metric values
        comparison_df: Output from compare_metrics_from_df()
        suffix1: Suffix for raw version (e.g., "_norm")
        suffix2: Suffix for transformed version (e.g., "_dblnorm")
        sort_by: Column to sort comparison_df by (e.g., "pearson_r")
    """
    # Sort metrics by similarity score (descending)
    sorted_metrics = comparison_df.sort_values(by=sort_by, ascending=False)['metric'].tolist()
    n_metrics = len(sorted_metrics)

    ncols = 8
    nrows = int(np.ceil(n_metrics * 2 / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 2.5))
    axs = axs.flatten()

    for i, metric in enumerate(sorted_metrics):
        col1 = f'{metric}{suffix1}'
        col2 = f'{metric}{suffix2}'

        if col1 not in all_events_df.columns or col2 not in all_events_df.columns:
            continue

        x = all_events_df[col1]
        y = all_events_df[col2]

        mask = ~x.isna() & ~y.isna()
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        # ---- 1. SCATTER DENSITY PLOT ----
        ax1 = axs[i * 2]
        
        if metric in ['3rd_ARR', '3rd_com', '3rd_maxpct', '3rd_w_peak']:
            bins=3
        elif metric in ['4th_w_peak']:
            bins=4
        elif metric in ['5th_w_peak']:
            bins=5
        else:
            bins=30        
        sns.histplot(x=x, y=y, bins=bins, pmax=0.9, ax=ax1, cmap="viridis", cbar=False)
        ax1.plot([x.min(), x.max()], [x.min(), x.max()], 'r--')  # 1:1 line
        ax1.set_title(f'{metric} (Raw vs DMC)')
        ax1.set_xlabel('Raw')
        ax1.set_ylabel('DMC')

        # ---- 2. RELATIVE DIFFERENCE HISTOGRAM ----
        ax2 = axs[i * 2 + 1]
        rel_diff = np.abs(x - y) / (np.abs(x) + 1e-8)
        ax2.hist(rel_diff, bins=30, color='skyblue', edgecolor='black')
        ax2.set_title(f'{metric} Rel. Diff.')
        ax2.set_xlabel('|x - y| / |x|')
        ax2.set_ylabel('Count')

    # Hide unused axes
    for j in range(n_metrics * 2, len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    # fig.savefig("sorted_metric_comparisons.png", dpi=300)
    plt.show()



def compare_metrics_from_df(df, suffix1, suffix2, is_robust_dict, tolerance=0.1):
    raw_suffixes = [suffix1]
    dmc_suffixes = [suffix2]
    suffixes = raw_suffixes + dmc_suffixes

    metric_bases = {
        col.replace(suffix, '') 
        for col in df.columns 
        for suffix in suffixes 
        if col.endswith(suffix)
    }

    results = []
    bland_altman_data = []

    # Initialize the scaler
#     scaler = StandardScaler()
    scaler = MinMaxScaler()

    for base in sorted(metric_bases):
        raw_col = next((base + sfx for sfx in raw_suffixes if base + sfx in df.columns), None)
        dmc_col = next((base + sfx for sfx in dmc_suffixes if base + sfx in df.columns), None)
        if not raw_col or not dmc_col:
            continue

        x, y = df[raw_col].values, df[dmc_col].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue

        # Scale the raw and DMC data to a range of [0, 1] using Min-Max normalization
        #x_scaled = x #scaler.fit_transform(x.reshape(-1, 1)).flatten()  # Reshape for scaler and flatten back
        #y_scaled = y #scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Scale the raw and DMC data to a range of [0, 1] using Min-Max normalization
        x_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten()  # Reshape for scaler and flatten back
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()        
        
        rel_diff = np.abs(x_scaled - y_scaled) / (np.abs(x_scaled) + 1e-8)
        abs_diff = y_scaled - x_scaled
        mean_xy = (x_scaled + y_scaled) / 2

        try:
            pearson_r_value = pearsonr(x_scaled, y_scaled)[0]
        except Exception:
            pearson_r_value = np.nan

        try:
            ttest_p = ttest_rel(x_scaled, y_scaled).pvalue
        except Exception:
            ttest_p = np.nan

        try:
            wilcoxon_p = wilcoxon(x_scaled, y_scaled).pvalue
        except Exception:
            wilcoxon_p = np.nan

        bias = abs_diff.mean()
        bias_std = abs_diff.std() / np.sqrt(len(x))  # standard error of the bias
        bias_conf_interval_lower = bias - 1.96 * bias_std
        bias_conf_interval_upper = bias + 1.96 * bias_std
        bias_conf_interval = (bias_conf_interval_lower, bias_conf_interval_upper)
        
        loa = 1.96 * abs_diff.std()
        loa_width = loa * 2  # width of the limits of agreement

        mean_val = mean_xy.mean()
        # is_robust = (np.abs(bias) <= 0.075 * (np.abs(mean_val) + 1e-8)) and ((rel_diff < tolerance).mean() >= 0.9)
        is_robust = 'royalblue' # is_robust_dict[base]
            
        results.append({
            "metric": base,
            "raw_column": raw_col,
            "dmc_column": dmc_col,
            "pearson_r": round(pearson_r_value, 3),
            "spearman_rho": round(spearmanr(x_scaled, y_scaled)[0], 3),
            "kendall_tau": round(kendalltau(x_scaled, y_scaled)[0], 3),
            "mae": round(mean_absolute_error(x_scaled, y_scaled), 4),
            "rmse": round(np.sqrt(mean_squared_error(x_scaled, y_scaled)), 4),
            "bias": round(bias, 4),
            "bias_conf_interval_lower": bias_conf_interval_lower,
            "bias_conf_interval_upper": bias_conf_interval_upper,
            "loa_width": round(loa_width, 4),
            "mean_rel_diff": round(rel_diff.mean(), 4),
            "std_rel_diff": round(rel_diff.std(), 4),
            f"agreement_within_{int(tolerance*100)}%": round((rel_diff < tolerance).mean() * 100, 1),
            "ttest_p": round(ttest_p, 4),
            "wilcoxon_p": round(wilcoxon_p, 4),
            "is_robust": is_robust,
        })

        bland_altman_data.append({
            'metric': base,
            'mean_xy': mean_xy,
            'abs_diff': abs_diff,
            'bias': bias,
            'loa': loa,
            'bias_conf_interval': bias_conf_interval,
            'bias_conf_interval_lower': bias_conf_interval_lower,
            'bias_conf_interval_upper': bias_conf_interval_upper,
            'pearson_r': pearson_r_value,
            'is_robust': is_robust
        })
    # First by robust (True first), then descending pearson_r
#     bland_altman_data_sorted = sorted(bland_altman_data, key=lambda x: (not x['is_robust'], -x['pearson_r']))  
    bland_altman_data_sorted = sorted(bland_altman_data, key=lambda x: (-x['pearson_r']))
    
    results_df = pd.DataFrame(results)    
    results_df.sort_values(by='pearson_r', ascending=False, inplace=True)
    results_df.reset_index(inplace=True, drop=True)        

    return results_df, bland_altman_data_sorted

def cluster_from_correlation1(corr_matrix, num_clusters=None, plot_dendrogram=True):

    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix)
    distance_matrix = np.clip(distance_matrix, 0, 1)  # Ensure all values are in [0,1]

    # Convert to condensed form for linkage
    condensed_dist = squareform(distance_matrix)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method='complete')

    # Plot dendrogram
    if plot_dendrogram:
        plt.figure(figsize=(15, 5))
        dendrogram(Z, labels=corr_matrix.columns, leaf_rotation=90)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    labels = fcluster(Z, num_clusters, criterion='maxclust')

    return pd.Series(labels, index=corr_matrix.columns, name='Cluster')

def cluster_from_correlation2(corr_matrix, method='average', num_clusters=None, plot_dendrogram=True):

    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Convert to condensed form for linkage
    condensed_dist = squareform(distance_matrix)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method='complete')

    # Plot dendrogram
    if plot_dendrogram:
        plt.figure(figsize=(10, 10))
        dendrogram(Z, labels=corr_matrix.columns, leaf_rotation=90)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.tight_layout()
        plt.show()

    # Get flat cluster labels
    labels = fcluster(Z, num_clusters, criterion='maxclust')
#     if num_clusters is not None:
#         labels = fcluster(Z, num_clusters, criterion='maxclust')
#     else:
#         labels = fcluster(Z, t=0.7, criterion='distance')  # adjustable threshold
    print(labels)
    return pd.Series(labels, index=corr_matrix.columns, name='Cluster')

def corr_and_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    pval_matrix = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)

    for i in range(n):
        for j in range(n):
            if i <= j:
                corr, pval = pearsonr(df[cols[i]], df[cols[j]])
                corr_matrix.iloc[i, j] = corr
                corr_matrix.iloc[j, i] = corr
                pval_matrix.iloc[i, j] = pval
                pval_matrix.iloc[j, i] = pval
    return corr_matrix, pval_matrix

def plot_bland_altman_grid(bland_altman_data, plots_per_row=3):
    n_metrics = len(bland_altman_data)
    ncols = plots_per_row
    nrows = -(-n_metrics // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, metric_data in zip(axes, bland_altman_data):
        mean_xy = metric_data['mean_xy']
        diff = metric_data['abs_diff']
        bias = metric_data['bias']
        loa = metric_data['loa']
        bias_conf_interval= metric_data['bias_conf_interval']
        bias_conf_interval_lower = metric_data['bias_conf_interval_lower']
        bias_conf_interval_upper = metric_data['bias_conf_interval_upper']        
        pearson_r_value = metric_data['pearson_r']
        is_robust = metric_data['is_robust']
        metric = metric_data['metric']

        color = is_robust
        ax.scatter(mean_xy, diff, alpha=0.1, edgecolors='k', linewidth=0.5, color=color)
        ax.axhline(bias, color='red', linestyle='--', label=f'Mean Bias = {bias:.2f}')
        ax.axhline(0, color='black', linestyle='-', label='Line of Equality')
        ax.axhline(bias + loa, color='gray', linestyle=':', label='Upper LoA')
        ax.axhline(bias - loa, color='gray', linestyle=':', label='Lower LoA')
#         ax.axhline(bias_conf_interval_lower, color='green', linestyle='--', label='95% CI of Bias')
#         ax.axhline(bias_conf_interval_upper, color='green', linestyle='--')
        ax.set_title(metric, fontsize=30)
        ax.set_xlabel('Mean of Raw and DMC', fontsize=20)
        ax.set_ylabel('DMC - Raw', fontsize=20)
        ax.legend(fontsize='small', loc='upper right')

    # Hide unused subplots
    for ax in axes[len(bland_altman_data):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    fig.savefig("Figures/DMC_Raw_BlandAltman2.png", facecolor='white', edgecolor='white', dpi=300)