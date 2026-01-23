

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