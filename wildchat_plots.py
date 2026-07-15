"""WildChat plotting functions. Each takes a WildChatMetrics instance."""

from scipy.stats import pearsonr, ttest_rel, ttest_ind, sem
from statsmodels.stats.multitest import multipletests as mult_test
from statsmodels.stats.proportion import proportions_ztest
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import csv
import os
from contextlib import redirect_stdout
from functools import partial
import polars as pl
from scipy.ndimage import uniform_filter1d

from wildchat_metrics import (
    WildChatMetrics, load_metrics, ensure_template_metrics,
    CUTOFF, VLINE_DATE, VERSIONS, OUTPUT_DIR,
    wildchat_color, model_colors, intent_map, domain_map, levels, trunc,
)


# Font sizes for all plots. Adjust these values to resize uniformly.
FONT_TICK = 16          # standard tick labels
FONT_ANNOT_SM = 14.5    # scatter-point label annotations
FONT_LABEL = 21         # axis labels, subplot titles, inline annotations (R)
FONT_TICK_PANEL = 22    # tick labels in dense multi-panel figures (quarters)
FONT_LABEL_PANEL = 24   # ylabels in multi-panel figures (intent/domain quarters)
FONT_ANNOT_LG = 26      # percent-change / diff annotations in trajectory plots
FONT_PANEL_HEADER = 30  # trajectory panel column titles, ylabels, compact suptitles
FONT_SUPTITLE = 43      # large figure super-titles (halves)


# ── Multi-dataset helpers ─────────────────────────────────────────────────────

def _plot_versions(m: WildChatMetrics):
    """Return [(dir_name, use_full, tsuffix)] for this dataset.

    All output dirs live under OUTPUT_DIR; non-4.8m datasets add a '{name}-plots/'
    subdir. Truncated version is omitted when the dataset has no cutoff.
    Falls back to the module-level VERSIONS for old caches without .dataset.
    """
    cfg = m.dataset
    if cfg is None:
        return VERSIONS
    has_cutoff = cfg.cutoff is not None
    sub = "" if cfg.name == "wildchat-4.8m" else f"{cfg.name}-plots/"
    full_dir = f"{OUTPUT_DIR}/{sub}wildchat-full"
    os.makedirs(full_dir, exist_ok=True)
    versions = [(full_dir, True, "")]
    if has_cutoff:
        trunc_dir = f"{OUTPUT_DIR}/{sub}wildchat-trunc"
        os.makedirs(trunc_dir, exist_ok=True)
        versions.append((trunc_dir, False, "_trunc"))
    return versions


def _ds_title(m: WildChatMetrics, use_full):
    """Return display title: 'WildChat-4.8M (Full)', 'WildChat-1M', etc."""
    _labels = {"wildchat-4.8m": "WildChat-4.8M", "wildchat-1m": "WildChat-1M"}
    cfg = m.dataset
    label = _labels.get(cfg.name, cfg.name) if cfg is not None else "WildChat-4.8M"
    has_cutoff = cfg is not None and cfg.cutoff is not None
    return label + (" (Full)" if (use_full and has_cutoff) else "")


def _show_vline(m: WildChatMetrics, use_full):
    """True only when showing full data for a dataset that has a known cutoff."""
    return use_full and m.dataset is not None and m.dataset.cutoff is not None


def _r_label(r_val, p_val):
    """Format R annotation with * if p < 0.05."""
    star = "^{\\!\\ast}" if p_val < 0.05 else ""
    return f"$R={r_val:.2f}{star}$"


def _date_ordinal(s):
    """Convert a Polars Date series to integer days since epoch for pearsonr."""
    return s.cast(pl.Int32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Time-series plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_activity_over_time(m: WildChatMetrics):
    k_fmt = FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}')

    for dir_name, use_full, tsuffix in _plot_versions(m):
        _conv_df = m.conv_by_day_df if use_full else trunc(m.conv_by_day_df)
        _user_df = m.user_by_day_df if use_full else trunc(m.user_by_day_df)

        conv_counts = _conv_df['conv_count'].to_numpy().astype(float)
        conv_smoothed = uniform_filter1d(conv_counts, size=14)
        user_counts = _user_df['user_count'].to_numpy().astype(float)
        user_smoothed = uniform_filter1d(user_counts, size=14)

        for log_scale in [False, True]:
            fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

            axes[0].scatter(_conv_df['date'], conv_counts, color=wildchat_color, alpha=0.2, s=16, zorder=2, marker='.')
            axes[0].plot(_conv_df['date'], conv_smoothed, "-", color=wildchat_color, linewidth=2, zorder=3)
            axes[0].set_ylabel('# Conversations', fontsize=FONT_LABEL)
            axes[0].yaxis.set_major_formatter(k_fmt)
            axes[0].tick_params(axis='both', labelsize=FONT_TICK)

            r = pearsonr(_date_ordinal(_conv_df['date']), _conv_df['conv_count'])
            print(f"Pearson correlation for conv count over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
            axes[0].text(0.02, 0.95, _r_label(r.statistic, r.pvalue), transform=axes[0].transAxes, ha="left", va="top", fontsize=FONT_LABEL)

            axes[1].scatter(_user_df['date'], user_counts, color=wildchat_color, alpha=0.2, s=16, zorder=2, marker='.')
            axes[1].plot(_user_df['date'], user_smoothed, "-", color=wildchat_color, linewidth=2, zorder=3)
            axes[1].set_ylabel('# Users', fontsize=FONT_LABEL)
            axes[1].yaxis.set_major_formatter(k_fmt)
            axes[1].tick_params(axis='both', labelsize=FONT_TICK)

            r = pearsonr(_date_ordinal(_user_df['date']), _user_df['user_count'])
            print(f"Pearson correlation for user count over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
            axes[1].text(0.02, 0.95, _r_label(r.statistic, r.pvalue), transform=axes[1].transAxes, ha="left", va="top", fontsize=FONT_LABEL)

            if log_scale:
                axes[0].set_yscale('log')
                axes[1].set_yscale('log')

            if _show_vline(m, use_full):
                for ax in axes:
                    ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
            axes[1].tick_params(axis='x', labelrotation=30)
            plt.setp(axes[1].get_xticklabels(), ha='right')

            axes[0].set_title(_ds_title(m, use_full), fontsize=FONT_LABEL)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            suffix = "_log" if log_scale else ""
            plt.savefig(f"{dir_name}/activity_over_time_conv_and_user_count{suffix}_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
            plt.close()


def plot_repeat_user_fraction(m: WildChatMetrics):
    for dir_name, use_full, tsuffix in _plot_versions(m):
        _df = m.repeat_by_day_df if use_full else trunc(m.repeat_by_day_df)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        dates = _df['date']
        means = _df['repeat_frac'].to_numpy()
        se = _df['repeat_se'].to_numpy()
        smoothed = uniform_filter1d(means, size=14)

        ax.errorbar(dates, means, yerr=se, fmt='.', color=wildchat_color, alpha=0.2, ms=4, elinewidth=0.5)
        ax.plot(dates, smoothed, '-', color=wildchat_color, linewidth=2)
        ax.set_ylabel('Repeat User Fraction', fontsize=FONT_LABEL)
        ax.set_title(_ds_title(m, use_full), fontsize=FONT_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
        ax.tick_params(axis='x', labelrotation=30)
        plt.setp(ax.get_xticklabels(), ha='right')

        if _show_vline(m, use_full):
            ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

        plt.savefig(f"{dir_name}/repeat_user_fraction_over_time_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_messages_and_sentence_length(m: WildChatMetrics):
    for dir_name, use_full, tsuffix in _plot_versions(m):
        _mssgs = m.mssgs_by_day_df if use_full else trunc(m.mssgs_by_day_df)
        _sent = m.sent_len_by_day_df if use_full else trunc(m.sent_len_by_day_df)

        fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

        msg_dates = _mssgs['date']
        msg_means = _mssgs['turn_count_mean'].to_numpy()
        msg_se = _mssgs['turn_count_se'].to_numpy()
        msg_smoothed = uniform_filter1d(msg_means, size=14)

        axes[0].errorbar(msg_dates, msg_means, yerr=msg_se, fmt='.', color=wildchat_color, alpha=0.2, ms=4, elinewidth=0.5)
        axes[0].plot(msg_dates, msg_smoothed, "-", color=wildchat_color, linewidth=2)
        axes[0].set_ylabel('Avg. Msgs. / Conv.', fontsize=FONT_LABEL)
        axes[0].tick_params(axis='both', labelsize=FONT_TICK)

        r = pearsonr(_date_ordinal(msg_dates), msg_means)
        print(f"Pearson correlation for avg messages/conv over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        if not use_full:
            axes[0].text(0.98, 0.02, _r_label(r.statistic, r.pvalue), transform=axes[0].transAxes, ha="right", va="bottom", fontsize=FONT_LABEL)
        else:
            axes[0].text(0.02, 0.02, _r_label(r.statistic, r.pvalue), transform=axes[0].transAxes, ha="left", va="bottom", fontsize=FONT_LABEL)

        sl_dates = _sent['date']
        sl_means = _sent['sent_len_mean'].to_numpy()
        sl_se = _sent['sent_len_se'].to_numpy()
        sl_smoothed = uniform_filter1d(sl_means, size=14)

        axes[1].errorbar(sl_dates, sl_means, yerr=sl_se, fmt='.', color=wildchat_color, alpha=0.2, ms=4, elinewidth=0.5)
        axes[1].plot(sl_dates, sl_smoothed, "-", color=wildchat_color, linewidth=2)
        axes[1].set_ylabel('Avg. Sent. Len. (en)', fontsize=FONT_LABEL)
        axes[1].tick_params(axis='both', labelsize=FONT_TICK)
        axes[1].set_ylim(5, 40)

        r = pearsonr(_date_ordinal(sl_dates), sl_means)
        print(f"Pearson correlation for avg sentence length over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        if not use_full:
            axes[1].text(0.98, 0.02, _r_label(r.statistic, r.pvalue), transform=axes[1].transAxes, ha="right", va="bottom", fontsize=FONT_LABEL)
        else:
            axes[1].text(0.02, 0.02, _r_label(r.statistic, r.pvalue), transform=axes[1].transAxes, ha="left", va="bottom", fontsize=FONT_LABEL)

        if _show_vline(m, use_full):
            for ax in axes:
                ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
        axes[1].tick_params(axis='x', labelrotation=30)
        plt.setp(axes[1].get_xticklabels(), ha='right')

        axes[0].set_title(_ds_title(m, use_full), fontsize=FONT_LABEL)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{dir_name}/activity_over_time_avg_mssgs_and_sent_len_wildchat{tsuffix}.pdf", bbox_inches='tight')


def plot_completion_over_time(m: WildChatMetrics):
    for dir_name, use_full, tsuffix in _plot_versions(m):
        _df = m.complete_day_df if use_full else trunc(m.complete_day_df)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        dates = _df['date']
        means = _df['completed_mean'].to_numpy()
        se = _df['completed_se'].to_numpy()
        smoothed = uniform_filter1d(means, size=14)

        ax.errorbar(dates, means, yerr=se, fmt='.', color=wildchat_color, alpha=0.2, ms=4, elinewidth=0.5)
        ax.plot(dates, smoothed, '-', color=wildchat_color, linewidth=2)
        ax.set_ylabel('Completion Rate', fontsize=FONT_LABEL)
        ax.set_title(_ds_title(m, use_full), fontsize=FONT_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
        ax.tick_params(axis='x', labelrotation=30)
        plt.setp(ax.get_xticklabels(), ha='right')

        if _show_vline(m, use_full):
            ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

        r = pearsonr(_date_ordinal(_df['date']), _df['completed_mean'])
        print(f"Pearson correlation for completion rate over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        ax.text(0.98, 0.02, _r_label(r.statistic, r.pvalue), transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_LABEL)

        plt.savefig(f"{dir_name}/complete_over_time_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


def _plot_intent_panel(m: WildChatMetrics, intents, tag, r_top_right=False):
    labels = [intent_map[c] for c in intents]
    n = len(intents)

    for dir_name, use_full, tsuffix in _plot_versions(m):
        _df = m.intent_day_mean_df if use_full else trunc(m.intent_day_mean_df)

        fig, axes = plt.subplots(n, 1, figsize=(5, 3.5 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for i, (col, label) in enumerate(zip(intents, labels)):
            ax = axes[i]

            dates = _df['date']
            y_vals = _df[col + '_mean'].to_numpy()
            y_err = _df[col + '_se'].to_numpy()
            smoothed = uniform_filter1d(y_vals, size=14)

            r = pearsonr(_date_ordinal(dates), y_vals)

            ax.errorbar(dates, y_vals, yerr=y_err, fmt='.', color=wildchat_color, alpha=0.2, ms=4, elinewidth=0.5)
            ax.plot(dates, smoothed, '-', color=wildchat_color, linewidth=2)

            ax.set_title(f"{label}", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

            if i == n - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
                ax.tick_params(axis='x', labelrotation=30)
                plt.setp(ax.get_xticklabels(), ha='right')
            ax.set_ylabel('% Tasks', fontsize=FONT_LABEL)

            if r_top_right:
                ypos, va = 0.97, 'top'
            else:
                ypos = 0.97 if i > 1 else 0.02
                va = 'top' if i > 1 else 'bottom'
            ax.text(0.98, ypos, _r_label(r.statistic, r.pvalue), transform=ax.transAxes, ha='right', va=va, fontsize=FONT_LABEL)

            print(f"Pearson correlation for {tag} intent ({col}) over time{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")

        if _show_vline(m, use_full):
            for ax in axes:
                ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

        fig.suptitle(_ds_title(m, use_full), fontsize=FONT_TICK_PANEL, y=0.94, x=0.52)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{dir_name}/intent_over_time_{tag}_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


_TOP4_INTENTS = ['INFORMATION_GATHERING', 'TEXT_GENERATION', 'INFORMATION_LOOKUP', 'WEB_SITE_NAVIGATION']
_OTHER_INTENTS = ['ANALYSIS', 'IMAGE_CREATION', 'OPEN_ENDED_DISCOVERY', 'SUMMARIZATION', 'TRANSLATION_OR_CONVERSION']


def plot_top4_intents(m: WildChatMetrics):
    _plot_intent_panel(m, _TOP4_INTENTS, 'top4')


def plot_other_intents(m: WildChatMetrics):
    _plot_intent_panel(m, _OTHER_INTENTS, 'other', r_top_right=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Activity-group plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_completion_by_activity_group(m: WildChatMetrics):
    for dir_name, _filt, tsuffix in m.ag_versions:
        complete_by_group_df = _filt.group_by('day_group').agg(
            pl.col("completed").mean().alias("completed_mean"),
            (pl.col("completed").std() / pl.len().sqrt()).alias("completed_se"),
        )

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.errorbar(complete_by_group_df['day_group'], complete_by_group_df['completed_mean'],
                     yerr=complete_by_group_df['completed_se'], fmt="o", color=wildchat_color, ms=4)
        ax.set_xlabel('# Days Active', fontsize=FONT_LABEL)
        ax.set_ylabel('Completion Rate', fontsize=FONT_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)

        r = pearsonr(complete_by_group_df['day_group'], complete_by_group_df['completed_mean'])
        print(f"Pearson correlation for completion by activity group{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        ax.text(0.02, 0.98, _r_label(r.statistic, r.pvalue), transform=ax.transAxes, ha="left", va="top", fontsize=FONT_LABEL)
        _title = _ds_title(m, tsuffix == '')
        ax.set_title(_title, fontsize=FONT_LABEL)
        plt.savefig(f"{dir_name}/complete_by_group_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_activity_by_activity_group(m: WildChatMetrics):
    for dir_name, _filt, tsuffix in m.ag_versions:
        fig, axs = plt.subplots(3, 1, figsize=(4.5, 10), sharex=True)

        # Row 0: Conversations per active day
        convs_per_date_per_ip = _filt.group_by(["day_group", "date", "hashed_ip"]).agg(pl.len().alias("convs_per_date_per_ip"))
        avg_convs_per_ip = convs_per_date_per_ip.group_by(["day_group", "hashed_ip"]).agg(pl.col("convs_per_date_per_ip").mean().alias("avg_convs_per_ip"))
        avg_convs_per_group = avg_convs_per_ip.group_by(["day_group"]).agg(
            pl.col("avg_convs_per_ip").mean().alias("avg_convs"),
            (pl.col("avg_convs_per_ip").std() / pl.len().sqrt()).alias("avg_convs_se"),
        ).sort("day_group")

        axs[0].errorbar(avg_convs_per_group['day_group'], avg_convs_per_group['avg_convs'],
                         yerr=avg_convs_per_group['avg_convs_se'], fmt="o", color=wildchat_color, ms=4)
        axs[0].set_ylabel('Conv. / Active Day', fontsize=FONT_LABEL)
        axs[0].tick_params(axis='both', labelsize=FONT_TICK)

        r = pearsonr(avg_convs_per_group['day_group'], avg_convs_per_group['avg_convs'])
        print(f"Pearson correlation for convs/active day by group{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        axs[0].text(0.02, 0.98, _r_label(r.statistic, r.pvalue), transform=axs[0].transAxes, ha="left", va="top", fontsize=FONT_LABEL)

        # Row 1: Messages per conversation
        avg_turns_per_ip = _filt.group_by(["day_group", "hashed_ip"]).agg(pl.col("turn").mean().alias("avg_turns_per_ip"))
        avg_turns_per_group = avg_turns_per_ip.group_by(["day_group"]).agg(
            pl.col("avg_turns_per_ip").mean().alias("avg_turns"),
            (pl.col("avg_turns_per_ip").std() / pl.len().sqrt()).alias("avg_turns_se"),
        ).sort("day_group")

        axs[1].errorbar(avg_turns_per_group['day_group'], avg_turns_per_group['avg_turns'],
                         yerr=avg_turns_per_group['avg_turns_se'], fmt="o", color=wildchat_color, ms=4)
        axs[1].set_ylabel('Messages / Conv.', fontsize=FONT_LABEL)
        axs[1].tick_params(axis='both', labelsize=FONT_TICK)

        r = pearsonr(avg_turns_per_group['day_group'], avg_turns_per_group['avg_turns'])
        print(f"Pearson correlation for messages/conv by group{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        axs[1].text(0.02, 0.98, _r_label(r.statistic, r.pvalue), transform=axs[1].transAxes, ha="left", va="top", fontsize=FONT_LABEL)

        # Row 2: Sentence length (English only)
        sent_len_per_ip = (
            _filt.filter(pl.col("language") == "English")
            .group_by(["day_group", "hashed_ip"])
            .agg(pl.col("mean_sentence_len_words").mean().alias("avg_sent_len_per_ip"))
        )
        avg_sent_len_per_group = sent_len_per_ip.group_by(["day_group"]).agg(
            pl.col("avg_sent_len_per_ip").mean().alias("avg_sent_len"),
            (pl.col("avg_sent_len_per_ip").std() / pl.len().sqrt()).alias("avg_sent_len_se"),
        ).sort("day_group")

        axs[2].errorbar(avg_sent_len_per_group['day_group'], avg_sent_len_per_group['avg_sent_len'],
                         yerr=avg_sent_len_per_group['avg_sent_len_se'], fmt="o", color=wildchat_color, ms=4)
        axs[2].set_ylabel('Sentence Length', fontsize=FONT_LABEL)
        axs[2].tick_params(axis='both', labelsize=FONT_TICK)

        r = pearsonr(avg_sent_len_per_group['day_group'], avg_sent_len_per_group['avg_sent_len'])
        print(f"Pearson correlation for sentence length by group{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
        axs[2].text(0.02, 0.98, _r_label(r.statistic, r.pvalue), transform=axs[2].transAxes, ha="left", va="top", fontsize=FONT_LABEL)

        axs[-1].set_xlabel("# Days Active", fontsize=FONT_LABEL)
        axs[0].set_title(_ds_title(m, tsuffix == ''), fontsize=FONT_LABEL)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(left=0.28)
        plt.savefig(f"{dir_name}/activity_group_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory halves & quarters
# ═══════════════════════════════════════════════════════════════════════════════

def plot_activity_trajectory_quarters(m: WildChatMetrics):
    activity_labels = {'turn': '# Msgs.\n/ Conv.', 'sent_len': 'Sent.\nLen.', 'completed': 'Compl. Rate'}
    activity_cols = ['turn', 'sent_len', 'completed']

    is_full_version = {m.ag_versions[0][0]: True}  # first entry is full

    for dir_name, _filt, tsuffix in m.ag_versions:
        is_full = dir_name in is_full_version

        quarter_df = (
            _filt.filter(pl.col('day_group') > 3)
            .with_columns(
                pl.when(pl.col('day_group') <= 10).then(pl.lit('low'))
                  .when(pl.col('day_group') <= 25).then(pl.lit('middle'))
                  .otherwise(pl.lit('high'))
                  .alias('level')
            )
            .with_columns(
                pl.when(pl.col('day_index') <= pl.col('day_group') / 4).then(1)
                  .when(pl.col('day_index') <= pl.col('day_group') / 2).then(2)
                  .when(pl.col('day_index') <= pl.col('day_group') * 3 / 4).then(3)
                  .otherwise(4)
                  .alias('quarter')
            )
        )

        per_ip_quarter = (
            quarter_df.group_by(['hashed_ip', 'level', 'quarter'])
            .agg([
                pl.col('turn').mean().alias('turn'),
                pl.col('mean_sentence_len_words').mean().alias('sent_len'),
                pl.col('completed').mean().alias('completed'),
            ])
        )

        q_activity_rate_dict = {col: {level: {q: [] for q in range(1, 5)} for level in levels} for col in activity_cols}

        for col in activity_cols:
            col_valid = per_ip_quarter.filter(pl.col(col).is_not_null())
            ip_quarter_counts = col_valid.group_by(['hashed_ip', 'level']).agg(pl.len().alias('n_quarters'))
            complete_ips = ip_quarter_counts.filter(pl.col('n_quarters') == 4).select(['hashed_ip', 'level'])
            col_complete = col_valid.join(complete_ips, on=['hashed_ip', 'level'], how='inner')

            for row in col_complete.iter_rows(named=True):
                q_activity_rate_dict[col][row['level']][row['quarter']].append(row[col])

        # Population
        dates = _filt['date']
        min_date, max_date = dates.min(), dates.max()
        total_days = (max_date - min_date).days
        q_boundaries = [min_date + datetime.timedelta(days=int(total_days * f)) for f in [0, 0.25, 0.5, 0.75, 1.0]]

        pop_quarter_df = _filt.with_columns(
            pl.when(pl.col('date') < q_boundaries[1]).then(1)
              .when(pl.col('date') < q_boundaries[2]).then(2)
              .when(pl.col('date') < q_boundaries[3]).then(3)
              .otherwise(4)
              .alias('pop_quarter')
        )

        pop_rate_dict = {col: {q: [] for q in range(1, 5)} for col in activity_cols}
        pop_daily = pop_quarter_df.group_by(['date', 'pop_quarter']).agg([
            pl.col('turn').mean().alias('turn'),
            pl.col('mean_sentence_len_words').mean().alias('sent_len'),
            pl.col('completed').mean().alias('completed'),
        ])
        for row in pop_daily.iter_rows(named=True):
            for col in activity_cols:
                if row[col] is not None:
                    pop_rate_dict[col][row['pop_quarter']].append(row[col])

        n_features = 3
        heights = ([[0.7, 0.38, 0.45, 0.4],
                    [0.32, 0.4, 0.6, 0.2],
                    [0.55, 0.6, 0.35, 0.4]] if is_full else
                   [[0.7, 0.27, 0.3, 0.3],
                    [0.12, 0.35, 0.55, 0.1],
                    [0.7, 0.7, 0.4, 0.25]])

        fig, axes = plt.subplots(nrows=n_features, ncols=4, figsize=(13, 2.75 * n_features), sharex=False,
                                 gridspec_kw={'hspace': 0, 'wspace': 0})

        for i, col in enumerate(activity_cols):
            axes[i][0].set_ylabel(activity_labels[col], fontsize=FONT_PANEL_HEADER)

            # Population column
            pop_corr = ttest_ind(pop_rate_dict[col][1], pop_rate_dict[col][4])
            randoms = [np.mean(pop_rate_dict[col][q]) for q in range(1, 5)]
            random_errs = [sem(pop_rate_dict[col][q]) for q in range(1, 5)]

            random_color = 'gray'
            if pop_corr.pvalue < 0.001 / (n_features * 4):
                random_color = wildchat_color

            all_raw_values = []
            for level in levels:
                for q in range(1, 5):
                    if len(q_activity_rate_dict[col][level][q]) > 0:
                        all_raw_values.append(np.mean(q_activity_rate_dict[col][level][q]))
            all_raw_values += randoms
            min_raw_value = min(all_raw_values)
            max_raw_value = max(all_raw_values)
            margin = (max_raw_value - min_raw_value) * 0.15

            random_pct = (randoms[-1] - randoms[0]) / randoms[0] * 100
            random_sign = '+' if random_pct > 0 else ''
            axes[i][3].errorbar(range(4), randoms, yerr=random_errs, fmt="-o", color=random_color, ms=10, linewidth=3, elinewidth=3)
            axes[i][3].set_ylim(min_raw_value - margin, max_raw_value + margin)
            axes[i][3].set_xlim(-0.75, 3.75)
            axes[i][3].yaxis.tick_right()
            axes[i][3].spines['left'].set_visible(False)
            q_pop_labels = [f"{q_boundaries[q].month}/{str(q_boundaries[q].year)[2:]}" for q in range(4)]
            if i == n_features - 1:
                axes[i][3].set_xticks(range(4), q_pop_labels, fontsize=FONT_TICK_PANEL, rotation=30)
            else:
                axes[i][3].set_xticks([])
            axes[i][3].tick_params(labelsize=FONT_TICK_PANEL)
            axes[i][3].text(0.5 if is_full else 0.7, heights[i][3], f"{random_sign}{random_pct:.1f}%",
                            transform=axes[i][3].transAxes, ha='center', va='bottom', fontsize=FONT_ANNOT_LG)

            if col == 'completed':
                axes[i][3].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

            for j, level in enumerate(levels):
                raw_corr = ttest_rel(q_activity_rate_dict[col][level][1], q_activity_rate_dict[col][level][4])

                raws = [np.mean(q_activity_rate_dict[col][level][q]) for q in range(1, 5)]
                raw_errs = [sem(q_activity_rate_dict[col][level][q]) for q in range(1, 5)]

                raw_color = 'gray'
                if raw_corr.pvalue < 0.05 / (n_features * 4):
                    raw_color = wildchat_color

                axes[i][j].errorbar(range(4), raws, yerr=raw_errs, fmt="-o", color=raw_color, ms=10, linewidth=3, elinewidth=3)
                axes[i][j].set_ylim(min_raw_value - margin, max_raw_value + margin)
                axes[i][j].set_xlim(-0.75, 3.75)

                if j > 0:
                    axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].set_xticks([])

                raw_pct = (raws[-1] - raws[0]) / raws[0] * 100
                raw_sign = '+' if raw_pct > 0 else ''
                axes[i][j].text(0.5, heights[i][j], f"{raw_sign}{raw_pct:.1f}%",
                                transform=axes[i][j].transAxes, ha='center', va='bottom', fontsize=FONT_ANNOT_LG)
                if i == n_features - 1:
                    axes[i][j].set_xticks(range(4), ['Q1', 'Q2', 'Q3', 'Q4'], fontsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_xticks([])
                if j == 0:
                    axes[i][j].tick_params(labelsize=FONT_TICK_PANEL)
                    if col == 'completed':
                        axes[i][j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
                else:
                    axes[i][j].set_yticks([])

                if i == 0:
                    axes[i][j].set_title(level.title(), fontsize=FONT_PANEL_HEADER)

            if i == 0:
                axes[i][3].set_title("Population", fontsize=FONT_PANEL_HEADER)

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(_ds_title(m, tsuffix == ''), ha='center', fontsize=FONT_PANEL_HEADER, y=0.98)
        plt.savefig(f"{dir_name}/activity_level_lifetime_changes_quarters_wildchat{tsuffix}.pdf", bbox_inches='tight')
        plt.close()


def plot_intent_trajectory_quarters(m: WildChatMetrics, annotate=True):
    """Intent frequency quarters — 4-column layout matching Bing styling."""
    int_to_clean = {
        'IMAGE_CREATION': 'Image\nCreation', 'OPEN_ENDED_DISCOVERY': 'Open-Ended\nDiscovery',
        'WEB_SITE_NAVIGATION': 'Website\nNavigation', 'INFORMATION_LOOKUP': 'Information\nLookup',
        'ANALYSIS': 'Analysis', 'TEXT_GENERATION': 'Text\nGeneration',
        'TRANSLATION_OR_CONVERSION': 'Translation or\nConversion',
        'INFORMATION_GATHERING': 'Information\nGathering', 'SUMMARIZATION': 'Summariz.',
    }
    n_features = len(m.intent_cols)

    annot_overrides = {
        ('WEB_SITE_NAVIGATION', 1): (0.15, 'bottom', 1.5),

    }

    for dir_name, _filt, tsuffix in m.ag_versions:
        # ── Build 4-quarter user data ──
        quarter_df = (
            _filt.filter(pl.col('day_group') > 3)
            .with_columns(
                pl.when(pl.col('day_group') <= 10).then(pl.lit('low'))
                  .when(pl.col('day_group') <= 25).then(pl.lit('middle'))
                  .otherwise(pl.lit('high'))
                  .alias('level')
            )
            .with_columns(
                pl.when(pl.col('day_index') <= pl.col('day_group') / 4).then(1)
                  .when(pl.col('day_index') <= pl.col('day_group') / 2).then(2)
                  .when(pl.col('day_index') <= pl.col('day_group') * 3 / 4).then(3)
                  .otherwise(4)
                  .alias('quarter')
            )
        )

        per_ip_quarter = (
            quarter_df.group_by(['hashed_ip', 'level', 'quarter'])
            .agg([pl.col(intent).mean().alias(intent) for intent in m.intent_cols])
        )

        q_intent_rate_dict = {intent: {level: {q: [] for q in range(1, 5)} for level in levels} for intent in m.intent_cols}
        for intent in m.intent_cols:
            col_valid = per_ip_quarter.filter(pl.col(intent).is_not_null())
            ip_quarter_counts = col_valid.group_by(['hashed_ip', 'level']).agg(pl.len().alias('n_quarters'))
            complete_ips = ip_quarter_counts.filter(pl.col('n_quarters') == 4).select(['hashed_ip', 'level'])
            col_complete = col_valid.join(complete_ips, on=['hashed_ip', 'level'], how='inner')
            for row in col_complete.iter_rows(named=True):
                q_intent_rate_dict[intent][row['level']][row['quarter']].append(row[intent])

        # ── Population quarters ──
        dates = _filt['date']
        min_date, max_date = dates.min(), dates.max()
        total_days = (max_date - min_date).days
        q_boundaries = [min_date + datetime.timedelta(days=int(total_days * f)) for f in [0, 0.25, 0.5, 0.75, 1.0]]

        pop_quarter_df = _filt.with_columns(
            pl.when(pl.col('date') < q_boundaries[1]).then(1)
              .when(pl.col('date') < q_boundaries[2]).then(2)
              .when(pl.col('date') < q_boundaries[3]).then(3)
              .otherwise(4)
              .alias('pop_quarter')
        )

        pop_daily = pop_quarter_df.group_by(['date', 'pop_quarter']).agg(
            [pl.col(intent).mean().alias(intent) for intent in m.intent_cols]
        )
        pop_rate_dict = {intent: {q: [] for q in range(1, 5)} for intent in m.intent_cols}
        for row in pop_daily.iter_rows(named=True):
            for intent in m.intent_cols:
                if row[intent] is not None:
                    pop_rate_dict[intent][row['pop_quarter']].append(row[intent])

        # ── Plot ──
        fig, axes = plt.subplots(nrows=n_features, ncols=4, figsize=(12, 2.75 * n_features), sharex=False,
                                 gridspec_kw={'hspace': 0, 'wspace': 0})

        q_pop_labels = [f"{q_boundaries[q].month}/{str(q_boundaries[q].year)[2:]}" for q in range(4)]

        for i, col in enumerate(m.intent_cols):
            axes[i][0].set_ylabel(int_to_clean[col], fontsize=FONT_LABEL_PANEL)

            # Gather all values for shared y-limits
            pop_corr = ttest_ind(pop_rate_dict[col][1], pop_rate_dict[col][4])
            randoms = [np.mean(pop_rate_dict[col][q]) for q in range(1, 5)]
            random_errs = [sem(pop_rate_dict[col][q]) for q in range(1, 5)]
            random_color = wildchat_color if pop_corr.pvalue < 0.001 / (n_features * 4) else 'gray'

            all_raw_values = []
            for level in levels:
                for q in range(1, 5):
                    if len(q_intent_rate_dict[col][level][q]) > 0:
                        all_raw_values.append(np.mean(q_intent_rate_dict[col][level][q]))
            all_raw_values += randoms
            min_raw = min(all_raw_values)
            max_raw = max(all_raw_values)
            margin = (max_raw - min_raw) * 0.15
            pct_decimals = 1 if (max_raw - min_raw) < 0.03 else 0

            # Population column
            random_pct = (randoms[-1] - randoms[0]) / randoms[0] * 100
            random_sign = '+' if random_pct > 0 else ''
            axes[i][3].errorbar(range(4), randoms, yerr=random_errs, fmt="-o", color=random_color, ms=10, linewidth=3, elinewidth=3)
            axes[i][3].set_ylim(min_raw - margin, max_raw + margin)
            axes[i][3].set_xlim(-0.75, 3.75)
            axes[i][3].yaxis.tick_right()
            axes[i][3].spines['left'].set_visible(False)
            axes[i][3].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=pct_decimals))
            axes[i][3].tick_params(labelsize=FONT_TICK_PANEL)
            if i == n_features - 1:
                axes[i][3].set_xticks(range(4), q_pop_labels, fontsize=FONT_TICK_PANEL, rotation=30)
            else:
                axes[i][3].set_xticks([])

            ycenter = (min_raw + max_raw) / 2
            ymid_pop = (min(randoms) + max(randoms)) / 2
            if (col, 3) in annot_overrides:
                ofactor, pop_va, pop_x = annot_overrides[(col, 3)]
                pop_y = (max(randoms) + margin * ofactor) if pop_va == 'bottom' else (min(randoms) - margin * ofactor)
            elif ymid_pop > ycenter:
                pop_y = min(randoms) - margin * 0.3
                pop_va = 'top'
                pop_x = 1.5
            else:
                pop_y = max(randoms) + margin * 0.3
                pop_va = 'bottom'
                pop_x = 1.5
            if annotate:
                axes[i][3].text(pop_x, pop_y, f"{random_sign}{random_pct:.1f}%",
                                ha='center', va=pop_va, fontsize=FONT_ANNOT_LG)

            # Activity-group columns
            for j, level in enumerate(levels):
                raw_corr = ttest_rel(q_intent_rate_dict[col][level][1], q_intent_rate_dict[col][level][4])
                raws = [np.mean(q_intent_rate_dict[col][level][q]) for q in range(1, 5)]
                raw_errs = [sem(q_intent_rate_dict[col][level][q]) for q in range(1, 5)]
                raw_color = wildchat_color if raw_corr.pvalue < 0.05 / (n_features * 4) else 'gray'

                axes[i][j].errorbar(range(4), raws, yerr=raw_errs, fmt="-o", color=raw_color, ms=10, linewidth=3, elinewidth=3)
                axes[i][j].set_ylim(min_raw - margin, max_raw + margin)
                axes[i][j].set_xlim(-0.75, 3.75)
                axes[i][j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=pct_decimals))
                if j > 0:
                    axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                if i == n_features - 1:
                    axes[i][j].set_xticks(range(4), ['Q1', 'Q2', 'Q3', 'Q4'], fontsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_xticks([])

                raw_pct = (raws[-1] - raws[0]) / raws[0] * 100
                raw_sign = '+' if raw_pct > 0 else ''
                ymid_raw = (min(raws) + max(raws)) / 2
                if (col, j) in annot_overrides:
                    ofactor, raw_va, raw_x = annot_overrides[(col, j)]
                    raw_y = (max(raws) + margin * ofactor) if raw_va == 'bottom' else (min(raws) - margin * ofactor)
                elif ymid_raw > ycenter:
                    raw_y = min(raws) - margin * 0.3
                    raw_va = 'top'
                    raw_x = 1.5
                else:
                    raw_y = max(raws) + margin * 0.3
                    raw_va = 'bottom'
                    raw_x = 1.5
                if annotate:
                    axes[i][j].text(raw_x, raw_y, f"{raw_sign}{raw_pct:.1f}%",
                                    ha='center', va=raw_va, fontsize=FONT_ANNOT_LG)
                if j == 0:
                    axes[i][j].tick_params(labelsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_yticks([])

                if i == 0:
                    axes[i][j].set_title(level.title(), fontsize=FONT_PANEL_HEADER)

            if i == 0:
                axes[i][3].set_title("Population", fontsize=FONT_PANEL_HEADER)

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(_ds_title(m, tsuffix == ''), ha='center', fontsize=FONT_PANEL_HEADER, y=0.915)
        plt.savefig(f"{dir_name}/intent_lifetime_change_quarters_wildchat{tsuffix}.pdf", bbox_inches='tight')
        plt.close()


def plot_completion_vs_intent_activity(m: WildChatMetrics):
    int_to_clean = {
        'IMAGE_CREATION': 'Image\nCreation', 'OPEN_ENDED_DISCOVERY': 'Open-Ended\nDiscovery',
        'WEB_SITE_NAVIGATION': 'Website\nNav.', 'INFORMATION_LOOKUP': 'Info.\nLookup',
        'ANALYSIS': 'Analysis', 'TEXT_GENERATION': 'Text\nGeneration',
        'TRANSLATION_OR_CONVERSION': 'Translation or Conversion',
        'INFORMATION_GATHERING': 'Information Gathering', 'SUMMARIZATION': 'Summarization',
    }

    label_offsets = {
        'OPEN_ENDED_DISCOVERY': (0, 5, 'left', 'bottom'),
        'IMAGE_CREATION':       (5, 5, 'left', 'top'),
        'WEB_SITE_NAVIGATION':  (5, 5, 'left', 'top'),
        'INFORMATION_LOOKUP':   (0, 5, 'right', 'top'),
        'TEXT_GENERATION':      (-5, 0, 'right', 'center'),
        'ANALYSIS':             (5, 0, 'left', 'center'),
        'TRANSLATION_OR_CONVERSION': (-5, 0, 'right', 'center'),
    }

    for dir_name, _filt, tsuffix in m.ag_versions:
        _intent_by_group = _filt.group_by('day_group').agg(
            [pl.col(c).mean().alias(f"{c}_mean") for c in m.intent_cols]
        ).sort('day_group')

        intent_group_dict = {}
        intent_group_pval_dict = {}
        for col in m.intent_cols:
            r = pearsonr(_intent_by_group['day_group'], _intent_by_group[col + '_mean'])
            print(f"Pearson correlation for intent ({col}) vs activity group{tsuffix}: R={r.statistic:.5f}, p={r.pvalue:.3g}")
            intent_group_dict[col] = r.statistic
            intent_group_pval_dict[col] = r.pvalue

        _intent_comp = _filt.group_by(['day_group', 'user_intent']).agg(
            pl.col("completed").mean().alias("completed_mean")
        )
        avg_intent_complete_dict = {}
        for col in m.intent_cols:
            avg_intent_complete_dict[col] = _intent_comp.filter(
                pl.col('user_intent') == col
            )['completed_mean'].mean()

        fig, ax = plt.subplots(figsize=(4.5, 5))
        ax.scatter(
            [avg_intent_complete_dict[c] * 100 for c in m.intent_cols],
            [intent_group_dict[c] for c in m.intent_cols],
            color=[wildchat_color if intent_group_pval_dict[c] < 1e-3 else "#a6a6a6" for c in m.intent_cols],
            s=50,
        )
        ax.axhline(y=0, color='black', linewidth=1)

        for col in m.intent_cols:
            if col in label_offsets:
                dx, dy, ha, va = label_offsets[col]
                ax.annotate(int_to_clean[col],
                            (avg_intent_complete_dict[col] * 100, intent_group_dict[col]),
                            textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, fontsize=FONT_ANNOT_SM)

        ax.set_xlim([60, 100])
        ax.set_ylabel("Pearson's $R$ Correlation\n(# Days Active vs. % Intent)", fontsize=FONT_LABEL)
        ax.set_xlabel("Avg. Completion %", fontsize=FONT_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_TICK)
        ax.tick_params(axis='x', labelsize=FONT_TICK)

        ax.set_title(_ds_title(m, tsuffix == ''), fontsize=FONT_LABEL)

        plt.savefig(f"{dir_name}/completed_vs_active_days_intent_prop_corr_wildchat{tsuffix}.pdf",
                     bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_domain_trajectory_quarters(m: WildChatMetrics, page=1):
    """Domain frequency quarters — 4-column layout matching intent quarters style."""
    dom_to_clean = {
        'ADULT': 'Adult', 'BIOLOGY': 'Biology',
        'BUSINESS_AND_FINANCE': 'Business &\nFinance',
        'COMPUTERS_AND_ELECTRONICS': 'Computers &\nElectronics',
        'CREATIVE_WRITING_AND_EDITING': 'Creat. Writ.\n& Editing',
        'DATA_ANALYSIS_AND_VISUALIZATION': 'Data Analys.\n& Vis.',
        'EDUCATION_AND_LEARNING': 'Education &\nLearning',
        'ENGINEERING_AND_DESIGN': 'Eng. &\nDesign',
        'ENTERTAINMENT': 'Entertain.', 'FASHION_AND_BEAUTY': 'Fashion\n& Beauty',
        'FOOD_AND_DRINK': 'Food &\nDrink', 'GAMING': 'Gaming',
        'HEALTH_AND_MEDICINE': 'Health &\nMedicine',
        'HISTORY_AND_CULTURE': 'History\n& Culture',
        'HOME_AND_AUTO': 'Home &\nAuto',
        'JOBS_AND_EMPLOYMENT': 'Jobs &\nEmployment',
        'LAW_AND_POLITICS': 'Law &\nPolitics',
        'MACHINE_LEARNING_AND_AI': 'ML & AI',
        'MARKETING_AND_SALES': 'Marketing\n& Sales',
        'MATHEMATICS_AND_LOGIC': 'Math.\n& Logic', 'OTHER': 'Other',
        'PHYSICS_AND_CHEMISTRY': 'Physics &\nChemistry',
        'PROFESSIONAL_WRITING_AND_EDITING': 'Prof. Writing\n& Editing',
        'PROGRAMMING_AND_SCRIPTING': 'Prog.\n& Scripting',
        'RELIGION_AND_PHILOSOPHY': 'Religion\n& Philos.',
        'SHOPPING_AND_ECOMMERCE': 'Shopping\n& eComm.',
        'SMALL_TALK_AND_CHATBOT': 'Small Talk\n& Chatbot',
        'SPORTS_AND_FITNESS': 'Sports &\nFitness',
        'TRANSLATION_AND_LANGUAGE': 'Translation\n& Language',
        'TRAVEL_AND_TOURISM': 'Travel &\nTourism',
    }

    all_domains = sorted(m.domain_cols)
    if 'OTHER' in all_domains:
        all_domains.remove('OTHER')
        all_domains.append('OTHER')
    if page == 1:
        cols = all_domains[:15]
    elif page == 2:
        cols = all_domains[15:]
    else:
        raise ValueError("Page must be 1 or 2.")   
    n_features = len(cols)

    for dir_name, _filt, tsuffix in m.ag_versions:
        # ── Build 4-quarter user data ──
        quarter_df = (
            _filt.filter(pl.col('day_group') > 3)
            .with_columns(
                pl.when(pl.col('day_group') <= 10).then(pl.lit('low'))
                  .when(pl.col('day_group') <= 25).then(pl.lit('middle'))
                  .otherwise(pl.lit('high'))
                  .alias('level')
            )
            .with_columns(
                pl.when(pl.col('day_index') <= pl.col('day_group') / 4).then(1)
                  .when(pl.col('day_index') <= pl.col('day_group') / 2).then(2)
                  .when(pl.col('day_index') <= pl.col('day_group') * 3 / 4).then(3)
                  .otherwise(4)
                  .alias('quarter')
            )
        )

        per_ip_quarter = (
            quarter_df.group_by(['hashed_ip', 'level', 'quarter'])
            .agg([pl.col(d).mean().alias(d) for d in cols])
        )

        q_domain_rate_dict = {d: {level: {q: [] for q in range(1, 5)} for level in levels} for d in cols}
        for d in cols:
            col_valid = per_ip_quarter.filter(pl.col(d).is_not_null())
            ip_quarter_counts = col_valid.group_by(['hashed_ip', 'level']).agg(pl.len().alias('n_quarters'))
            complete_ips = ip_quarter_counts.filter(pl.col('n_quarters') == 4).select(['hashed_ip', 'level'])
            col_complete = col_valid.join(complete_ips, on=['hashed_ip', 'level'], how='inner')
            for row in col_complete.iter_rows(named=True):
                q_domain_rate_dict[d][row['level']][row['quarter']].append(row[d])

        # ── Population quarters ──
        dates = _filt['date']
        min_date, max_date = dates.min(), dates.max()
        total_days = (max_date - min_date).days
        q_boundaries = [min_date + datetime.timedelta(days=int(total_days * f)) for f in [0, 0.25, 0.5, 0.75, 1.0]]

        pop_quarter_df = _filt.with_columns(
            pl.when(pl.col('date') < q_boundaries[1]).then(1)
              .when(pl.col('date') < q_boundaries[2]).then(2)
              .when(pl.col('date') < q_boundaries[3]).then(3)
              .otherwise(4)
              .alias('pop_quarter')
        )

        pop_daily = pop_quarter_df.group_by(['date', 'pop_quarter']).agg(
            [pl.col(d).mean().alias(d) for d in cols]
        )
        pop_rate_dict = {d: {q: [] for q in range(1, 5)} for d in cols}
        for row in pop_daily.iter_rows(named=True):
            for d in cols:
                if row[d] is not None:
                    pop_rate_dict[d][row['pop_quarter']].append(row[d])

        # ── Plot ──
        fig, axes = plt.subplots(nrows=n_features, ncols=4, figsize=(12, 2.75 * n_features), sharex=False,
                                 gridspec_kw={'hspace': 0, 'wspace': 0})

        q_pop_labels = [f"{q_boundaries[q].month}/{str(q_boundaries[q].year)[2:]}" for q in range(4)]

        for i, col in enumerate(cols):
            axes[i][0].set_ylabel(dom_to_clean.get(col, col), fontsize=FONT_LABEL_PANEL)

            pop_corr = ttest_ind(pop_rate_dict[col][1], pop_rate_dict[col][4])
            randoms = [np.mean(pop_rate_dict[col][q]) for q in range(1, 5)]
            random_errs = [sem(pop_rate_dict[col][q]) for q in range(1, 5)]
            random_color = wildchat_color if pop_corr.pvalue < 0.001 / (n_features * 4) else 'gray'

            all_raw_values = []
            for level in levels:
                for q in range(1, 5):
                    if len(q_domain_rate_dict[col][level][q]) > 0:
                        all_raw_values.append(np.mean(q_domain_rate_dict[col][level][q]))
            all_raw_values += randoms
            min_raw = min(all_raw_values)
            max_raw = max(all_raw_values)
            margin = (max_raw - min_raw) * 0.15
            pct_decimals = 1 if (max_raw - min_raw) < 0.03 else 0

            # Population column
            axes[i][3].errorbar(range(4), randoms, yerr=random_errs, fmt="-o", color=random_color, ms=10, linewidth=3, elinewidth=3)
            axes[i][3].set_ylim(min_raw - margin, max_raw + margin)
            axes[i][3].set_xlim(-0.75, 3.75)
            axes[i][3].yaxis.tick_right()
            axes[i][3].spines['left'].set_visible(False)
            axes[i][3].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=pct_decimals))
            axes[i][3].tick_params(labelsize=FONT_TICK_PANEL)
            if i == n_features - 1:
                axes[i][3].set_xticks(range(4), q_pop_labels, fontsize=FONT_TICK_PANEL, rotation=30)
            else:
                axes[i][3].set_xticks([])

            # Activity-group columns
            for j, level in enumerate(levels):
                raw_corr = ttest_rel(q_domain_rate_dict[col][level][1], q_domain_rate_dict[col][level][4])
                raws = [np.mean(q_domain_rate_dict[col][level][q]) for q in range(1, 5)]
                raw_errs = [sem(q_domain_rate_dict[col][level][q]) for q in range(1, 5)]
                raw_color = wildchat_color if raw_corr.pvalue < 0.05 / (n_features * 4) else 'gray'

                axes[i][j].errorbar(range(4), raws, yerr=raw_errs, fmt="-o", color=raw_color, ms=10, linewidth=3, elinewidth=3)
                axes[i][j].set_ylim(min_raw - margin, max_raw + margin)
                axes[i][j].set_xlim(-0.75, 3.75)
                axes[i][j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=pct_decimals))
                if j > 0:
                    axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                if i == n_features - 1:
                    axes[i][j].set_xticks(range(4), ['Q1', 'Q2', 'Q3', 'Q4'], fontsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_xticks([])
                if j == 0:
                    axes[i][j].tick_params(labelsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_yticks([])

                if i == 0:
                    axes[i][j].set_title(level.title(), fontsize=FONT_PANEL_HEADER)

            if i == 0:
                axes[i][3].set_title("Population", fontsize=FONT_PANEL_HEADER)

        # Manual y-tick override for CREATIVE_WRITING_AND_EDITING to avoid overlap.
        cwe_idx = next((i for i, c in enumerate(cols) if c == 'CREATIVE_WRITING_AND_EDITING'), None)
        if cwe_idx is not None:
            for k in [0, 3]:
                axes[cwe_idx][k].set_yticks([0.10, 0.20, 0.30])

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(_ds_title(m, tsuffix == ''), ha='center', fontsize=FONT_PANEL_HEADER, y=0.9)
        plt.savefig(f"{dir_name}/domain_lifetime_change_quarters_wildchat_{page}{tsuffix}.pdf", bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Templated / API-like usage plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_template_over_time(m: WildChatMetrics):
    """Plot templated-conversation % and count over time (scatter + 14-day smooth)."""
    ensure_template_metrics(m)
    k_fmt = FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}')

    for dir_name, use_full, tsuffix in _plot_versions(m):
        _df = m.template_by_day_df if use_full else trunc(m.template_by_day_df)

        dates = _df['date']
        pcts = _df['template_pct'].to_numpy()
        counts = _df['template_count'].to_numpy().astype(float)
        pcts_smooth = uniform_filter1d(pcts, size=14)
        counts_smooth = uniform_filter1d(counts, size=14)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)

        # Panel configs: (data, smoothed, ylabel, log_scale, formatter)
        panels = [
            (pcts, pcts_smooth, '% Template Conv.', False, None),
            (counts, counts_smooth, '# Template Conv.', False, k_fmt),
            (counts, counts_smooth, '# Template Conv.', True, k_fmt),
        ]

        for ax, (vals, smooth, ylabel, log_scale, yfmt) in zip(axes, panels):
            ax.scatter(dates, vals, color=wildchat_color, alpha=0.2, s=16, zorder=2, marker='.')
            ax.plot(dates, smooth, '-', color=wildchat_color, linewidth=2, zorder=3)
            ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
            ax.tick_params(axis='both', labelsize=FONT_TICK)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4 if use_full else 3))
            ax.tick_params(axis='x', labelrotation=30)
            plt.setp(ax.get_xticklabels(), ha='right')

            if yfmt:
                ax.yaxis.set_major_formatter(yfmt)
            if log_scale:
                ax.set_yscale('log')
            if _show_vline(m, use_full):
                ax.axvline(VLINE_DATE, color='black', linestyle='--', linewidth=1)

        axes[1].set_title(_ds_title(m, use_full), fontsize=FONT_LABEL, pad=12)
        plt.tight_layout()
        plt.savefig(f"{dir_name}/template_over_time_wildchat{tsuffix}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


def _latex_escape(s):
    """Escape special LaTeX characters and replace problematic Unicode."""
    for char in ['\\', '&', '%', '$', '#', '_', '{', '}', '~', '^']:
        s = s.replace(char, '\\' + char)
    s = s.replace('→', '$\\rightarrow$')
    s = s.replace('←', '$\\leftarrow$')
    return s


def print_template_table_latex(
    m: WildChatMetrics,
    k=20,
    display_len=80,
    out_dir=OUTPUT_DIR,
    translations=None,
):
    """Generate a standalone LuaLaTeX table of the k most common templates.

    Writes a standalone .tex file that renders with full Unicode support
    (Cyrillic, CJK, Vietnamese, emoji, etc.) when compiled with LuaLaTeX.

    On Overleaf: upload template_table.tex, set the compiler to LuaLaTeX.

    Include in your main document with:
        \\usepackage{graphicx}
        \\begin{table*}[ht]
        \\centering
        \\caption{Top k most common conversation templates.}
        \\label{tab:templates}
        \\includegraphics{template_table.pdf}
        \\end{table*}

    Args:
        m: WildChatMetrics instance (must have template_groups_df populated).
        k: Number of templates to include.
        display_len: Number of characters of the fingerprint to show.
        out_dir: Directory to write the .tex file and sparklines.
        translations: Dict mapping 1-based row number to language label, e.g.
            {4: "Russian", 6: "Vietnamese"}. Adds a translation placeholder
            row below each specified row.

    Returns:
        Path to the generated .tex file.
    """
    from pathlib import Path
    from scipy.ndimage import uniform_filter1d

    ensure_template_metrics(m)
    groups = m.template_groups_df.head(k)
    total_convos = m.template_by_day_df["total_count"].sum()

    out_path = Path(out_dir)
    spark_dir = out_path / "template_sparklines"
    spark_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate sparkline PDFs ---
    # Build a full date range for consistent x-axes across all sparklines
    all_dates = m.template_by_day_df["date"].to_list()
    date_min, date_max = min(all_dates), max(all_dates)
    import datetime
    full_dates = [date_min + datetime.timedelta(days=d)
                  for d in range((date_max - date_min).days + 1)]

    daily_by_fp = getattr(m, "template_daily_by_fp_df", None)

    for i, row in enumerate(groups.iter_rows(named=True)):
        rank = i + 1
        fp = row["fingerprint"]
        spark_path = spark_dir / f"spark_{rank}.pdf"

        if daily_by_fp is not None:
            fp_daily = daily_by_fp.filter(pl.col("fingerprint") == fp)
            date_to_count = dict(zip(fp_daily["date"].to_list(), fp_daily["count"].to_list()))
        else:
            date_to_count = {}

        counts = np.array([date_to_count.get(d, 0) for d in full_dates], dtype=float)
        smoothed = uniform_filter1d(counts, size=14)

        fig, ax = plt.subplots(figsize=(2, 1))
        ax.plot(full_dates, smoothed, color=wildchat_color, linewidth=1)
        ax.fill_between(full_dates, smoothed, alpha=0.15, color=wildchat_color)
        ax.axvline(datetime.date(CUTOFF.year, CUTOFF.month, CUTOFF.day),
                   color='black', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xlim(full_dates[0], full_dates[-1])
        ax.axis("off")
        fig.savefig(spark_path, bbox_inches="tight", pad_inches=0.01, dpi=150)
        plt.close(fig)

    print(f"Wrote {k} sparklines to {spark_dir}/")

    # --- Build the standalone .tex document ---
    lines = []
    lines.append(r"\documentclass[border=2pt]{standalone}")
    lines.append(r"\usepackage{fontspec}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{emoji}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\setmainfont{Noto Serif}")
    lines.append(r"\begin{document}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\extrarowheight}{4pt}")
    lines.append(r"\begin{tabular}{r>{\raggedright\arraybackslash}m{10cm}crr}")
    lines.append(r"\toprule")
    lines.append(r"\# & \bfseries Template Prefix & \bfseries Trend & \bfseries Count & \bfseries \% \\")
    lines.append(r"\midrule")

    for i, row in enumerate(groups.iter_rows(named=True)):
        rank = i + 1
        raw = row.get("original_prefix") or row["fingerprint"]
        prefix = raw.replace("<| start user message |>", "").lstrip()
        prefix = prefix[:display_len].replace("\n", " ").strip()
        prefix = _latex_escape(prefix)
        prefix += r"\dotfill"
        count = f"{row['count']:,}"
        pct = f"{row['count'] / total_convos * 100:.2f}"
        spark_inc = r"$\vcenter{\hbox{\includegraphics[height=3em]{template_sparklines/spark_" + str(rank) + r"}}}$"
        lines.append(f"{rank} & {prefix} & {spark_inc} & {count} & {pct}\\% \\\\")

        # Add translation placeholder if requested
        if translations and rank in translations:
            lang = _latex_escape(translations[rank])
            lines.append(f"& \\textit{{[{lang}] Translation:}} \\dotfill & & & \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")

    tex = "\n".join(lines)

    out_path = Path(out_dir)
    tex_file = out_path / "template_table.tex"
    tex_file.write_text(tex, encoding="utf-8")
    print(f"Wrote {tex_file}")

    return str(tex_file)


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════════════════════════════

def _export_csvs_for_version(filt_df, m, suffix, out_dir):
    """Export intent-completion, domain-prevalence, and intent-prevalence CSVs for one AG version."""
    low_df_raw = filt_df.filter(pl.col('day_group') < 11)
    middle_df_raw = filt_df.filter((pl.col('day_group') > 10) & (pl.col('day_group') < 26))
    high_df_raw = filt_df.filter(pl.col('day_group') > 25)

    # ── Intent completion by group ──
    mid_pvals, high_pvals = [], []
    for intent in m.intent_cols:
        low_intent = low_df_raw.filter(pl.col('user_intent') == intent)
        mid_intent = middle_df_raw.filter(pl.col('user_intent') == intent)
        high_intent = high_df_raw.filter(pl.col('user_intent') == intent)

        low_count, low_n = low_intent['completed'].sum(), len(low_intent)
        mid_count, mid_n = mid_intent['completed'].sum(), len(mid_intent)
        high_count, high_n = high_intent['completed'].sum(), len(high_intent)

        _, p_mid = proportions_ztest([low_count, mid_count], [low_n, mid_n], alternative='two-sided')
        _, p_high = proportions_ztest([low_count, high_count], [low_n, high_n], alternative='two-sided')
        mid_pvals.append(p_mid)
        high_pvals.append(p_high)

    intent_comp_df = filt_df.group_by(['day_group', 'user_intent']).agg(
        pl.col('completed').mean().alias('completed_mean')
    )
    n_tests = 2 * len(m.intent_cols)

    fname = f'{out_dir}/wildchat-intent-completion-by-group{suffix}.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['intent', 'low', 'middle', 'high', 'p_mid', 'p_high'])
        for i, intent in enumerate(m.intent_cols):
            low_val = intent_comp_df.filter((pl.col('user_intent') == intent) & (pl.col('day_group') < 11))['completed_mean'].mean()
            mid_val = intent_comp_df.filter((pl.col('user_intent') == intent) & (pl.col('day_group') > 10) & (pl.col('day_group') < 26))['completed_mean'].mean()
            high_val = intent_comp_df.filter((pl.col('user_intent') == intent) & (pl.col('day_group') > 25))['completed_mean'].mean()
            writer.writerow([intent, low_val, mid_val, high_val,
                             mid_pvals[i] * n_tests, high_pvals[i] * n_tests])
    print(f"Saved {fname}")

    # ── Domain prevalence by group ──
    domain_by_group_raw = filt_df.group_by('day_group').agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in m.domain_cols]
    ).sort('day_group')

    dom_mid_pvals, dom_high_pvals = [], []
    for domain in m.domain_cols:
        low_count_d = int(low_df_raw[domain].sum())
        low_n_d = len(low_df_raw)
        mid_count_d = int(middle_df_raw[domain].sum())
        mid_n_d = len(middle_df_raw)
        high_count_d = int(high_df_raw[domain].sum())
        high_n_d = len(high_df_raw)

        _, p_mid_d = proportions_ztest([low_count_d, mid_count_d], [low_n_d, mid_n_d], alternative='two-sided')
        _, p_high_d = proportions_ztest([low_count_d, high_count_d], [low_n_d, high_n_d], alternative='two-sided')
        dom_mid_pvals.append(p_mid_d)
        dom_high_pvals.append(p_high_d)

    dom_n_tests = 2 * len(m.domain_cols)

    low_group = domain_by_group_raw.filter(pl.col('day_group') < 11)
    mid_group = domain_by_group_raw.filter((pl.col('day_group') > 10) & (pl.col('day_group') < 26))
    high_group = domain_by_group_raw.filter(pl.col('day_group') > 25)

    fname = f'{out_dir}/wildchat-domain-prevalence-by-group{suffix}.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'low', 'middle', 'high', 'p_mid', 'p_high'])
        for i, domain in enumerate(m.domain_cols):
            low_val = low_group[domain + '_mean'].mean()
            mid_val = mid_group[domain + '_mean'].mean()
            high_val = high_group[domain + '_mean'].mean()
            writer.writerow([domain, low_val, mid_val, high_val,
                             dom_mid_pvals[i] * dom_n_tests, dom_high_pvals[i] * dom_n_tests])
    print(f"Saved {fname}")

    # ── Intent prevalence by group ──
    intent_by_group_raw = filt_df.group_by('day_group').agg(
        [pl.col(c).mean().alias(f"{c}_mean") for c in m.intent_cols]
    ).sort('day_group')

    int_prev_mid_pvals, int_prev_high_pvals = [], []
    for intent in m.intent_cols:
        low_count_i = int(low_df_raw[intent].sum())
        low_n_i = len(low_df_raw)
        mid_count_i = int(middle_df_raw[intent].sum())
        mid_n_i = len(middle_df_raw)
        high_count_i = int(high_df_raw[intent].sum())
        high_n_i = len(high_df_raw)

        _, p_mid_i = proportions_ztest([low_count_i, mid_count_i], [low_n_i, mid_n_i], alternative='two-sided')
        _, p_high_i = proportions_ztest([low_count_i, high_count_i], [low_n_i, high_n_i], alternative='two-sided')
        int_prev_mid_pvals.append(p_mid_i)
        int_prev_high_pvals.append(p_high_i)

    int_prev_n_tests = 2 * len(m.intent_cols)

    low_intent_prev = intent_by_group_raw.filter(pl.col('day_group') < 11)
    mid_intent_prev = intent_by_group_raw.filter((pl.col('day_group') > 10) & (pl.col('day_group') < 26))
    high_intent_prev = intent_by_group_raw.filter(pl.col('day_group') > 25)

    fname = f'{out_dir}/wildchat-intent-prevalence-by-group{suffix}.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['intent', 'low', 'middle', 'high', 'p_mid', 'p_high'])
        for i, intent in enumerate(m.intent_cols):
            low_val = low_intent_prev[intent + '_mean'].mean()
            mid_val = mid_intent_prev[intent + '_mean'].mean()
            high_val = high_intent_prev[intent + '_mean'].mean()
            writer.writerow([intent, low_val, mid_val, high_val,
                             int_prev_mid_pvals[i] * int_prev_n_tests,
                             int_prev_high_pvals[i] * int_prev_n_tests])
    print(f"Saved {fname}")


def export_csvs(m: WildChatMetrics, out_dir=OUTPUT_DIR):
    for _, filt_df, tsuffix in m.ag_versions:
        _export_csvs_for_version(filt_df, m, tsuffix, out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Unique category count trajectory (quarters)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_num_unique_categories_quarters(m: WildChatMetrics):
    """# unique intents and # unique domains per user-quarter, by activity level."""
    row_configs = [
        ("# Unique\nIntents", "user_intent"),
        ("# Unique\nDomains", "conversation_domain"),
    ]

    for dir_name, _filt, tsuffix in m.ag_versions:
        # Build quarter assignments (need day_group > 3 for 4 non-empty quarters)
        quarter_df = (
            _filt.filter(pl.col('day_group') > 3)
            .with_columns(
                pl.when(pl.col('day_group') <= 10).then(pl.lit('low'))
                  .when(pl.col('day_group') <= 25).then(pl.lit('middle'))
                  .otherwise(pl.lit('high'))
                  .alias('level')
            )
            .with_columns(
                pl.when(pl.col('day_index') <= pl.col('day_group') / 4).then(1)
                  .when(pl.col('day_index') <= pl.col('day_group') / 2).then(2)
                  .when(pl.col('day_index') <= pl.col('day_group') * 3 / 4).then(3)
                  .otherwise(4)
                  .alias('quarter')
            )
        )

        # For each row, compute per-user per-quarter # unique categories
        all_row_dicts = []
        for _, col_name in row_configs:
            unique_counts = (
                quarter_df.group_by(['hashed_ip', 'level', 'quarter'])
                .agg(pl.col(col_name).n_unique().alias('n_unique'))
            )
            # Keep only users who have data in all 4 quarters
            ip_qcounts = unique_counts.group_by(['hashed_ip', 'level']).agg(pl.len().alias('nq'))
            complete_ips = ip_qcounts.filter(pl.col('nq') == 4).select(['hashed_ip', 'level'])
            unique_counts = unique_counts.join(complete_ips, on=['hashed_ip', 'level'], how='inner')

            rate_dict = {level: {q: [] for q in range(1, 5)} for level in levels}
            for row in unique_counts.iter_rows(named=True):
                rate_dict[row['level']][row['quarter']].append(row['n_unique'])
            all_row_dicts.append(rate_dict)

        n_rows = len(row_configs)
        heights = [[0.25, 0.6, 0.6],
                    [0.2, 0.55, 0.63]]

        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(12, 2.75 * n_rows),
                                 sharex=False,
                                 gridspec_kw={'hspace': 0, 'wspace': 0})

        for i, (ylabel, _) in enumerate(row_configs):
            rate_dict = all_row_dicts[i]
            axes[i][0].set_ylabel(ylabel, fontsize=FONT_PANEL_HEADER)

            # Compute shared y-limits across all 3 levels
            all_raw_values = []
            for level in levels:
                for q in range(1, 5):
                    if len(rate_dict[level][q]) > 0:
                        all_raw_values.append(np.mean(rate_dict[level][q]))
            min_raw_value = min(all_raw_values)
            max_raw_value = max(all_raw_values)

            for j, level in enumerate(levels):
                raw_corr = ttest_rel(rate_dict[level][1], rate_dict[level][4])

                raws = [np.mean(rate_dict[level][q]) for q in range(1, 5)]
                raw_errs = [sem(rate_dict[level][q]) for q in range(1, 5)]

                raw_color = 'gray'
                if raw_corr.pvalue < 0.01:
                    raw_color = wildchat_color

                axes[i][j].errorbar(range(4), raws, yerr=raw_errs, fmt="-o",
                                    color=raw_color, ms=10, linewidth=3, elinewidth=3)
                axes[i][j].set_ylim(min_raw_value - 0.4, max_raw_value + 0.4)
                axes[i][j].set_xlim(-0.75, 3.75)

                # Remove vertical splines
                if j > 0:
                    axes[i][j].spines['left'].set_visible(False)
                if j < 2:
                    axes[i][j].spines['right'].set_visible(False)

                diff = raws[-1] - raws[0]
                sign = '+' if diff > 0 else ''
                axes[i][j].text(0.5, heights[i][j], f"{sign}{diff:.2f}",
                                transform=axes[i][j].transAxes, ha='center', va='bottom', fontsize=FONT_ANNOT_LG)

                if i == n_rows - 1:
                    axes[i][j].set_xticks(range(4), ['Q1', 'Q2', 'Q3', 'Q4'], fontsize=FONT_TICK_PANEL)
                else:
                    axes[i][j].set_xticks([])

                if j != 0:
                    axes[i][j].set_yticks([])
                else:
                    axes[i][j].tick_params(axis='y', labelsize=FONT_TICK_PANEL)

                if i == 0:
                    axes[i][j].set_title(level.title(), fontsize=FONT_ANNOT_LG)

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.suptitle(_ds_title(m, tsuffix == ''), ha='center', fontsize=FONT_ANNOT_LG, y=1.01)
        plt.savefig(f"{dir_name}/num_unique_categories_lifetime_change_quarters_wildchat{tsuffix}.pdf",
                     bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Load WildChat metrics once, then generate all figures, stats, and CSV exports."
    )
    parser.add_argument(
        "--dataset", default="wildchat-4.8m",
        help="Dataset key (see wildchat_metrics.DATASETS). Default: wildchat-4.8m.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force a fresh metrics load instead of using the on-disk cache.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Metrics loading is the expensive step; do it once and reuse for every plot.
    m = load_metrics(dataset=args.dataset, use_cache=not args.no_cache)

    # Core figures (mirrors the active cells of wildchat_fast.ipynb).
    steps = [
        ("Activity over time", plot_activity_over_time),
        ("Repeat-user fraction", plot_repeat_user_fraction),
        ("Messages and sentence length", plot_messages_and_sentence_length),
        ("Completion over time", plot_completion_over_time),
        ("Top-4 intents", plot_top4_intents),
        ("Other intents", plot_other_intents),
        ("Completion by activity group", plot_completion_by_activity_group),
        ("Activity by activity group", plot_activity_by_activity_group),
        ("Activity trajectory (quarters)", plot_activity_trajectory_quarters),
        ("Intent trajectory (quarters)", partial(plot_intent_trajectory_quarters, annotate=False)),
        ("Domain trajectory (quarters, page 1)", partial(plot_domain_trajectory_quarters, page=1)),
        ("Domain trajectory (quarters, page 2)", partial(plot_domain_trajectory_quarters, page=2)),
        ("Completion vs. intent activity", plot_completion_vs_intent_activity),
        ("Unique category counts (quarters)", plot_num_unique_categories_quarters),
        ("Template usage over time", plot_template_over_time),
    ]

    # LaTeX template table (.tex written to disk; compile with LuaLaTeX, e.g. on Overleaf).
    steps.append((
        "Template table (LaTeX)",
        partial(print_template_table_latex, k=10, display_len=250,
                translations={4: "Russian", 6: "Vietnamese", 9: "French"}),
    ))

    # CSV summaries.
    steps.append(("CSV export", export_csvs))

    for label, fn in steps:
        print(f"\n{label}")
        fn(m)

    print("\nDone.")


if __name__ == "__main__":
    main()

