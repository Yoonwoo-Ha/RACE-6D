"""
Plotting utilities to visualize training logs.
"""

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path, PurePath


COCO_EVAL_FIELDS = {
    'mAP': 0,       # AP @ IoU=0.50:0.95
    'AP@50': 1,     # AP @ IoU=0.50
    'AP@75': 2,     # AP @ IoU=0.75
    'AP@S': 3,      # AP @ small
    'AP@M': 4,      # AP @ medium
    'AP@L': 5,      # AP @ large
    'AR@1': 6,      # AR @ maxDets=1
    'AR@10': 7,     # AR @ maxDets=10
    'AR@100': 8,    # AR @ maxDets=100
    'AR@S': 9,      # AR @ small
    'AR@M': 10,     # AR @ medium
    'AR@L': 11,     # AR @ large
}

BOP_EVAL_FIELDS = {
    "bop_vsd",
    "bop_mssd",
    "bop_mspd",
    "bop_ar",
}


_ACC_BASES = [
    "adds_acc",
    "pck_0.05",
    "bbox_ciou_acc",
    "zoom_ciou_acc",
    "adds_rot_acc",
    "score_med",
]
_ACC_SUFFIXES = ["_aux_0", "_aux_1", "_enc_0"]
HIGHER_IS_BETTER = [f"{b}{s}" for b in _ACC_BASES for s in _ACC_SUFFIXES]
HIGHER_IS_BETTER.extend(["cls_accuracy", "gt_recall"])
HIGHER_IS_BETTER.extend(COCO_EVAL_FIELDS.keys())
HIGHER_IS_BETTER.extend(BOP_EVAL_FIELDS)
HIGHER_IS_BETTER.extend(
    [
        "cls_auc",
        "adds_auc",
        "adds_coarse_auc",
        "keypoint_auc",
        "coarse_keypoint_auc",
        "metric_pck5",
        "pck@5px",
        "pck@10px",
        "pck@20px",
        "metric_mask_iou",
        "metric_region_acc"
    ]
)

LOWER_IS_BETTER_EXTRA = [
    "metric_cls_error",
    "metric_tz_err_mm",
    "metric_rot_err_deg",
    "metric_uv_err_px",
    "metric_kpt_err_px",
    "metric_bbox_wh_err_px",
    "mean_kpt_error_px",
    "median_kpt_error_px",
]


def _parse_log_field(field):
    """Return the metric name and an optional explicit train/test split."""
    for split in ("train", "test"):
        prefix = f"{split}_"
        if field.startswith(prefix):
            return field[len(prefix):], split
    return field, None


def plot_logs(
    logs,
    fields=("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col=0,
    log_name="log.txt",
):
    func_name = "plot_utils.py::plot_logs"

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
                Expect list[Path] or single Path obj, received {type(logs)}"
            )

    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument:\n{dir}"
            )
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}. Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(24, 8))

    if len(fields) == 1:
        axs = [axs]

    fig.subplots_adjust(wspace=0.4, hspace=0.3)

    base_acc_fields = [
        "adds_acc",
        "pck_0.05",
        "bbox_ciou_acc",
        "zoom_ciou_acc",
        "adds_rot_acc",
        "score_med",
    ]
    layer_suffixes = ["_aux_0", "_aux_1", "_enc_0"]
    higher_is_better = [
        f"{base}{layer}" for base in base_acc_fields for layer in layer_suffixes
    ]
    higher_is_better.extend(["cls_accuracy", "gt_recall"])
    higher_is_better.extend(COCO_EVAL_FIELDS.keys())
    higher_is_better.extend(BOP_EVAL_FIELDS)
    higher_is_better.extend(
        [
            "cls_auc",
            "adds_auc",
            "adds_coarse_auc",
            "keypoint_auc",
            "coarse_keypoint_auc",
            "metric_pck5",
            "pck@5px",
            "pck@10px",
            "pck@20px",
            "metric_mask_iou",
        ]
    )

    lower_is_better_extra = [
        "metric_cls_error",
        "metric_tz_err_mm",
        "metric_rot_err_deg",
        "metric_uv_err_px",
        "metric_kpt_err_px",
        "metric_bbox_wh_err_px",
        "mean_kpt_error_px",
        "median_kpt_error_px",
    ]

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field in COCO_EVAL_FIELDS:
                if (
                    "test_coco_eval_bbox" in df.columns
                    and not df.test_coco_eval_bbox.isna().all()
                ):
                    coco_eval = (
                        pd.DataFrame(
                            np.stack(df.test_coco_eval_bbox.dropna().values)[:, COCO_EVAL_FIELDS[field]]
                        )
                        .ewm(com=ewm_col)
                        .mean()
                    )
                    axs[j].plot(
                        coco_eval, c=color, linestyle="-", marker="o", markersize=2.5
                    )

                    best_idx = coco_eval[0].idxmax()
                    best_value = coco_eval[0].iloc[best_idx]
                    last_idx = coco_eval[0].index[-1]
                    last_value = coco_eval[0].iloc[-1]

                    axs[j].axhline(
                        y=best_value,
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=1.5,
                    )

                    y_min, y_max = axs[j].get_ylim()
                    y_range = y_max - y_min
                    stack_offset = y_range * 0.06

                    axs[j].annotate(
                        f"best {best_value:.4f} (ep {best_idx})",
                        xy=(0, best_value),
                        xytext=(0.02, best_value + stack_offset / 4),
                        textcoords=("axes fraction", "data"),
                        fontsize=10,
                        va="bottom",
                        ha="left",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="lightgreen",
                            ec="green",
                            alpha=0.8,
                        ),
                    )

                    if last_idx != best_idx:
                        axs[j].axhline(
                            y=last_value,
                            color="royalblue",
                            linestyle="--",
                            alpha=0.7,
                            linewidth=1.5,
                        )
                        axs[j].annotate(
                            f"last {last_value:.4f} (ep {last_idx})",
                            xy=(0, last_value),
                            xytext=(0.02, last_value - stack_offset / 2),
                            textcoords=("axes fraction", "data"),
                            fontsize=10,
                            va="top",
                            ha="left",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="lightskyblue",
                                ec="royalblue",
                                alpha=0.8,
                            ),
                        )
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
                df_interpolated = df.interpolate()
                df_ewm = df_interpolated.ewm(com=ewm_col).mean()

                metric_field, explicit_split = _parse_log_field(field)
                is_higher_better = metric_field in higher_is_better
                is_lower_better = not is_higher_better and (
                    metric_field.startswith("loss_")
                    or metric_field.startswith("metric_")
                    or metric_field in lower_is_better_extra
                )

                train_field = field if explicit_split == "train" else f"train_{field}"
                if (
                    explicit_split != "test"
                    and train_field in df_ewm.columns
                ):
                    train_plot_values = df_ewm[train_field]
                    axs[j].plot(
                        df_ewm.index,
                        train_plot_values,
                        color="blue",
                        linestyle="-",
                        marker="o",
                        markersize=2.5,
                        label="Train",
                    )

                test_field = field if explicit_split == "test" else f"test_{field}"
                if (
                    explicit_split != "train"
                    and test_field in df_ewm.columns
                    and not df_ewm[test_field].isna().all()
                ):
                    test_values = df_ewm[test_field]

                    axs[j].plot(
                        df_ewm.index,
                        test_values,
                        color="red",
                        linestyle="--",
                        marker="o",
                        markersize=2.5,
                        label="Test",
                    )

                    # Best test value annotation (last epoch on tie)
                    valid_test = df_ewm[test_field].dropna()
                    if len(valid_test) > 0:
                        if is_higher_better:
                            best_val = valid_test.max()
                            line_color, box_color = "green", "lightgreen"
                        elif is_lower_better:
                            best_val = valid_test.min()
                            line_color, box_color = "orange", "moccasin"
                        else:
                            best_val = None

                        if best_val is not None:
                            best_idx = valid_test[valid_test == best_val].index[-1]
                            last_idx = valid_test.index[-1]
                            last_val = valid_test.iloc[-1]
                            best_above = last_idx == best_idx or best_val >= last_val

                            axs[j].axhline(
                                y=best_val, color=line_color,
                                linestyle="--", alpha=0.7, linewidth=1.5,
                            )
                            y_min, y_max = axs[j].get_ylim()
                            offset = (y_max - y_min) * 0.015
                            axs[j].annotate(
                                f"{best_val:.4f} (ep {best_idx})",
                                xy=(0, best_val),
                                xytext=(0.02, best_val + (offset if best_above else -offset)),
                                textcoords=("axes fraction", "data"),
                                fontsize=10,
                                va="bottom" if best_above else "top",
                                ha="left",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc=box_color, ec=line_color, alpha=0.8,
                                ),
                            )

                            if last_idx != best_idx:
                                axs[j].axhline(
                                    y=last_val, color="royalblue",
                                    linestyle="--", alpha=0.7, linewidth=1.5,
                                )
                                axs[j].annotate(
                                    f"last {last_val:.4f} (ep {last_idx})",
                                    xy=(0, last_val),
                                    xytext=(0.02, last_val + (-offset if best_above else offset)),
                                    textcoords=("axes fraction", "data"),
                                    fontsize=10,
                                    va="top" if best_above else "bottom",
                                    ha="left",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        fc="lightskyblue", ec="royalblue", alpha=0.8,
                                    ),
                                )

    for ax, field in zip(axs, fields):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best")

        ax.set_title(field)

        if ax.get_ylim()[1] - ax.get_ylim()[0] > 0:
            current_ymin, current_ymax = ax.get_ylim()
            margin = (current_ymax - current_ymin) * 0.15
            ax.set_ylim(current_ymin - margin, current_ymax + margin)

        metric_field, _ = _parse_log_field(field)
        if metric_field in BOP_EVAL_FIELDS:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        elif metric_field in higher_is_better:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3)


def plot_logs_2x(
    logs,
    fields=("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col=0,
    log_name="log.txt",
):
    func_name = "plot_utils.py::plot_logs"

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
                Expect list[Path] or single Path obj, received {type(logs)}"
            )

    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument:\n{dir}"
            )
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}. Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(24, 8))

    if len(fields) == 1:
        axs = [axs]

    fig.subplots_adjust(wspace=0.4, hspace=0.3)

    higher_is_better = HIGHER_IS_BETTER
    lower_is_better_extra = LOWER_IS_BETTER_EXTRA

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field in COCO_EVAL_FIELDS:
                if (
                    "test_coco_eval_bbox" in df.columns
                    and not df.test_coco_eval_bbox.isna().all()
                ):
                    coco_eval = (
                        pd.DataFrame(
                            np.stack(df.test_coco_eval_bbox.dropna().values)[:, COCO_EVAL_FIELDS[field]]
                        )
                        .ewm(com=ewm_col)
                        .mean()
                    )
                    axs[j].plot(
                        coco_eval, c=color, linestyle="-", marker="o", markersize=2.5
                    )

                    best_idx = coco_eval[0].idxmax()
                    best_value = coco_eval[0].iloc[best_idx]
                    last_idx = coco_eval[0].index[-1]
                    last_value = coco_eval[0].iloc[-1]

                    axs[j].axhline(
                        y=best_value,
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        linewidth=1.5,
                    )

                    y_min, y_max = axs[j].get_ylim()
                    y_range = y_max - y_min
                    stack_offset = y_range * 0.06

                    axs[j].annotate(
                        f"best {best_value:.4f} (ep {best_idx})",
                        xy=(0, best_value),
                        xytext=(0.02, best_value + stack_offset / 4),
                        textcoords=("axes fraction", "data"),
                        fontsize=10,
                        va="bottom",
                        ha="left",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="lightgreen",
                            ec="green",
                            alpha=0.8,
                        ),
                    )

                    if last_idx != best_idx:
                        axs[j].axhline(
                            y=last_value,
                            color="royalblue",
                            linestyle="--",
                            alpha=0.7,
                            linewidth=1.5,
                        )
                        axs[j].annotate(
                            f"last {last_value:.4f} (ep {last_idx})",
                            xy=(0, last_value),
                            xytext=(0.02, last_value - stack_offset / 2),
                            textcoords=("axes fraction", "data"),
                            fontsize=10,
                            va="top",
                            ha="left",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="lightskyblue",
                                ec="royalblue",
                                alpha=0.8,
                            ),
                        )
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
                df_interpolated = df.interpolate()
                df_ewm = df_interpolated.ewm(com=ewm_col).mean()

                metric_field, explicit_split = _parse_log_field(field)
                is_higher_better = metric_field in higher_is_better
                is_lower_better = not is_higher_better and (
                    metric_field.startswith("loss_")
                    or metric_field.startswith("metric_")
                    or metric_field in lower_is_better_extra
                )
                train_field = field if explicit_split == "train" else f"train_{field}"
                if (
                    explicit_split != "test"
                    and train_field in df_ewm.columns
                ):
                    train_plot_values = df_ewm[train_field]
                    axs[j].plot(
                        df_ewm.index,
                        train_plot_values,
                        color="blue",
                        linestyle="-",
                        marker="o",
                        markersize=2.5,
                        label="Train",
                    )

                test_field = field if explicit_split == "test" else f"test_{field}"
                if (
                    explicit_split != "train"
                    and test_field in df_ewm.columns
                    and not df_ewm[test_field].isna().all()
                ):
                    test_values = df_ewm[test_field]

                    axs[j].plot(
                        df_ewm.index,
                        test_values,
                        color="red",
                        linestyle="--",
                        marker="o",
                        markersize=2.5,
                        label="Test",
                    )

                    # Best test value annotation (last epoch on tie)
                    valid_test = df_ewm[test_field].dropna()
                    if len(valid_test) > 0:
                        if is_higher_better:
                            best_val = valid_test.max()
                            line_color, box_color = "green", "lightgreen"
                        elif is_lower_better:
                            best_val = valid_test.min()
                            line_color, box_color = "orange", "moccasin"
                        else:
                            best_val = None

                        if best_val is not None:
                            best_idx = valid_test[valid_test == best_val].index[-1]
                            last_idx = valid_test.index[-1]
                            last_val = valid_test.iloc[-1]
                            best_above = last_idx == best_idx or best_val >= last_val

                            axs[j].axhline(
                                y=best_val, color=line_color,
                                linestyle="--", alpha=0.7, linewidth=1.5,
                            )
                            y_min, y_max = axs[j].get_ylim()
                            y_offset = (y_max - y_min) * 0.015
                            axs[j].annotate(
                                f"{best_val:.4f} (ep {best_idx})",
                                xy=(0, best_val),
                                xytext=(0.02, best_val + (y_offset if best_above else -y_offset)),
                                textcoords=("axes fraction", "data"),
                                fontsize=10,
                                va="bottom" if best_above else "top",
                                ha="left",
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc=box_color, ec=line_color, alpha=0.8,
                                ),
                            )

                            if last_idx != best_idx:
                                axs[j].axhline(
                                    y=last_val, color="royalblue",
                                    linestyle="--", alpha=0.7, linewidth=1.5,
                                )
                                axs[j].annotate(
                                    f"last {last_val:.4f} (ep {last_idx})",
                                    xy=(0, last_val),
                                    xytext=(0.02, last_val + (-y_offset if best_above else y_offset)),
                                    textcoords=("axes fraction", "data"),
                                    fontsize=10,
                                    va="top" if best_above else "bottom",
                                    ha="left",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        fc="lightskyblue", ec="royalblue", alpha=0.8,
                                    ),
                                )

    for ax, field in zip(axs, fields):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best")

        ax.set_title(field)

        if ax.get_ylim()[1] - ax.get_ylim()[0] > 0:
            current_ymin, current_ymax = ax.get_ylim()
            margin = (current_ymax - current_ymin) * 0.15
            ax.set_ylim(current_ymin - margin, current_ymax + margin)

        metric_field, _ = _parse_log_field(field)
        if metric_field in BOP_EVAL_FIELDS:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        elif metric_field in higher_is_better:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3)


def _metric_direction(metric_field):
    """Return 1 if a larger value is better, -1 if a smaller one is, 0 if unknown."""
    if metric_field in HIGHER_IS_BETTER:
        return 1
    if (
        metric_field.startswith("loss_")
        or metric_field.startswith("metric_")
        or metric_field in LOWER_IS_BETTER_EXTRA
    ):
        return -1
    return 0


def _split_log_runs(df):
    """Split a log into runs where the epoch counter restarts at 0.

    A resumed run that repeats earlier epochs is merged with its predecessor,
    the later rows winning. A restart at epoch 0 (another run appended to the
    same log file) becomes a separate run.
    """
    if "epoch" not in df.columns:
        return [df]
    epochs = pd.to_numeric(df["epoch"], errors="coerce").ffill().fillna(0).astype(int)
    runs, start = [], 0
    for i in range(1, len(df) + 1):
        if i == len(df) or epochs.iloc[i] <= epochs.iloc[i - 1]:
            segment = df.iloc[start:i]
            if runs and int(epochs.iloc[start]) > 0:
                merged = pd.concat([runs[-1], segment])
                runs[-1] = merged[~merged["epoch"].duplicated(keep="last")].sort_values("epoch")
            else:
                runs.append(segment)
            start = i
    return [run.reset_index(drop=True) for run in runs]


def _run_series(df, metric_field, split):
    """Epoch-indexed values of one metric and split, or None if the run did not log it."""
    if metric_field in COCO_EVAL_FIELDS:
        if split != "test" or "test_coco_eval_bbox" not in df.columns:
            return None
        rows = df.test_coco_eval_bbox.dropna()
        if rows.empty:
            return None
        values = np.stack(rows.values)[:, COCO_EVAL_FIELDS[metric_field]]
        series = pd.Series(values, index=rows.index, dtype=float)
    else:
        column = f"{split}_{metric_field}"
        if column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return None
    epochs = df["epoch"] if "epoch" in df.columns else pd.Series(df.index, index=df.index)
    series.index = epochs.loc[series.index].astype(int).values
    return series


def plot_logs_vs(
    logs,
    fields=("mAP", "bop_ar", "metric_tz_error", "metric_rot_error"),
    labels=None,
    splits=("train", "test"),
    ewm_col=0,
    log_name="log.txt",
    max_cols=4,
):
    """Overlay the epoch curves of several runs (e.g. RGB vs RGB-D) on shared axes.

    Each run gets one color. Test curves are solid and train curves dashed, so a
    train/test gap shows up per run. On the test curve, a star marks the best
    value and a circle the last one; both values are written in the legend, so
    runs of different length stay comparable.

    Args:
        logs: Run directories (or log files), as Path or str.
        fields: Metric names without the split prefix (``"metric_tz_error"``),
            COCO fields (``"mAP"``, ``"AP@50"``, ...) or BOP fields
            (``"bop_ar"``, ...). A ``train_``/``test_`` prefix plots that split only.
        labels: Legend names of the runs. Defaults to the directory names.
        splits: Splits drawn for fields without a prefix. COCO and BOP fields
            exist for the test split only.
        ewm_col: Exponential smoothing (``com``); 0 draws raw values.
        log_name: Log file name inside each run directory.
        max_cols: Maximum subplot columns.

    A field that a run did not log is skipped for that run. A log in which the
    epoch counter restarts at 0 is drawn as several runs (``label #2``, ...);
    epochs repeated by a resumed run keep their latest values.

    Returns:
        ``(fig, axs)``.
    """
    if isinstance(logs, (str, PurePath)):
        logs = [logs]
    paths = []
    for log in logs:
        path = Path(log)
        path = path / log_name if path.is_dir() else path
        if not path.is_file():
            raise ValueError(f"plot_logs_vs - missing log file: {path}")
        paths.append(path)
    if labels is None:
        labels = [path.parent.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("logs and labels must have equal lengths")
    fields = list(fields)
    if not fields:
        raise ValueError("fields must not be empty")

    dfs, run_labels = [], []
    for path, label in zip(paths, labels):
        runs = _split_log_runs(pd.read_json(path, lines=True))
        if len(runs) > 1:
            print(f"plot_logs_vs info: {path} holds {len(runs)} runs (epoch restarts at 0); "
                  f"later ones are labelled '{label} #2', ...")
        for k, run in enumerate(runs):
            dfs.append(run)
            run_labels.append(label if k == 0 else f"{label} #{k + 1}")
    colors = sns.color_palette(n_colors=len(dfs))
    ncols = min(max_cols, len(fields))
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False
    )

    for ax, field in zip(axs.flat, fields):
        metric_field, explicit_split = _parse_log_field(field)
        direction = _metric_direction(metric_field)
        field_splits = [explicit_split] if explicit_split else list(splits)
        for df, label, color in zip(dfs, run_labels, colors):
            for split in field_splits:
                series = _run_series(df, metric_field, split)
                if series is None:
                    continue
                series = series.ewm(com=ewm_col).mean()
                is_main = split == "test" or explicit_split == "train"
                text = f"{label} ({split})"
                if is_main and direction:
                    best_epoch = series.idxmax() if direction > 0 else series.idxmin()
                    best_value = series.loc[best_epoch]
                    last_epoch, last_value = series.index[-1], series.iloc[-1]
                    text += (
                        f"  last {last_value:.4f} (ep {last_epoch})"
                        f" | best {best_value:.4f} (ep {best_epoch})"
                    )
                    ax.plot(best_epoch, best_value, marker="*", markersize=13,
                            color=color, markeredgecolor="black", zorder=4)
                    ax.plot(last_epoch, last_value, marker="o", markersize=7,
                            color=color, markeredgecolor="black", zorder=4)
                ax.plot(
                    series.index,
                    series.values,
                    color=color,
                    linestyle="-" if is_main else "--",
                    linewidth=2.0 if is_main else 1.3,
                    alpha=1.0 if is_main else 0.75,
                    marker="o",
                    markersize=2.5,
                    label=text,
                )

        ax.set_title(field)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        if metric_field in BOP_EVAL_FIELDS:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        elif metric_field in HIGHER_IS_BETTER:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for ax in axs.flat[len(fields):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axs


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(
        files, sns.color_palette("Blues", n_colors=len(files)), names
    ):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs
