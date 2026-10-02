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

    _acc_bases = [
        "adds_acc",
        "pck_0.05",
        "bbox_ciou_acc",
        "zoom_ciou_acc",
        "adds_rot_acc",
        "score_med",
    ]
    _acc_suffixes = ["_aux_0", "_aux_1", "_enc_0"]
    higher_is_better = [f"{b}{s}" for b in _acc_bases for s in _acc_suffixes]
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
            "metric_region_acc"
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


def _load_vs_logs(logs, labels=None, groups=None):
    """Load JSON-lines comparison logs and attach display metadata."""
    if not isinstance(logs, (list, tuple)) or not logs:
        raise ValueError("logs must be a non-empty list of JSON-lines paths")

    paths = [Path(path) for path in logs]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"comparison log does not exist: {path}")

    if labels is None:
        labels = [path.stem for path in paths]
    if groups is None:
        groups = ["all"] * len(paths)
    if len(labels) != len(paths) or len(groups) != len(paths):
        raise ValueError("logs, labels, and groups must have equal lengths")

    frames = []
    for path, label, group in zip(paths, labels, groups):
        frame = pd.read_json(path, lines=True)
        frame["_vs_label"] = str(label)
        frame["_vs_group"] = str(group)
        frame["_vs_path"] = str(path)
        frames.append(frame)
    return frames


def plot_logs_vs(
    logs,
    fields,
    labels=None,
    groups=None,
    reducer="mean",
    max_cols=4,
):
    """Compare aggregate log fields as grouped side-by-side bars.

    Unlike :func:`plot_logs_2x`, this helper compares independent evaluation
    logs instead of epoch curves. Each input file may contain one row per
    image; ``reducer`` collapses those rows before plotting.

    Args:
        logs: JSON-lines file paths.
        fields: Numeric columns to compare.
        labels: Legend labels, e.g. ``["FP", "INT8", "FP", "INT8"]``.
        groups: X-axis groups, e.g.
            ``["No post", "No post", "Post TopK", "Post TopK"]``.
        reducer: ``"mean"`` or ``"median"``.
        max_cols: Maximum subplot columns.

    Returns:
        ``(fig, axs, summary_df)``.
    """
    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'")
    frames = _load_vs_logs(logs, labels=labels, groups=groups)
    fields = list(fields)
    if not fields:
        raise ValueError("fields must not be empty")

    rows = []
    for frame in frames:
        for field in fields:
            if field not in frame:
                raise KeyError(
                    f"missing field '{field}' in {frame['_vs_path'].iloc[0]}"
                )
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            value = (
                values.mean() if reducer == "mean" else values.median()
            )
            rows.append(
                {
                    "field": field,
                    "group": frame["_vs_group"].iloc[0],
                    "label": frame["_vs_label"].iloc[0],
                    "value": float(value),
                    "images": int(len(values)),
                }
            )
    summary = pd.DataFrame(rows)

    unique_groups = list(dict.fromkeys(summary["group"]))
    unique_labels = list(dict.fromkeys(summary["label"]))
    ncols = min(max_cols, len(fields))
    nrows = int(np.ceil(len(fields) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    palette = dict(
        zip(unique_labels, sns.color_palette(n_colors=len(unique_labels)))
    )
    x = np.arange(len(unique_groups), dtype=np.float64)
    width = 0.8 / max(len(unique_labels), 1)

    for field, ax in zip(fields, axs.flat):
        field_rows = summary[summary["field"] == field]
        for label_index, label in enumerate(unique_labels):
            values = []
            for group in unique_groups:
                match = field_rows[
                    (field_rows["group"] == group)
                    & (field_rows["label"] == label)
                ]
                values.append(
                    float(match["value"].iloc[0]) if len(match) else np.nan
                )
            offset = (
                label_index - (len(unique_labels) - 1) * 0.5
            ) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=palette[label],
                alpha=0.85,
            )
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(field)
        ax.set_xticks(x, unique_groups)
        ax.grid(axis="y", alpha=0.2)

    for ax in axs.flat[len(fields):]:
        ax.set_visible(False)
    handles, legend_labels = axs.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncols=len(unique_labels),
            frameon=False,
        )
    fig.suptitle(f"Log comparison ({reducer})", y=1.01)
    fig.tight_layout()
    return fig, axs, summary


def plot_logs_distribution_vs(
    logs,
    fields,
    labels=None,
    groups=None,
    kind="ecdf",
    bins=40,
    clip_quantile=0.99,
):
    """Overlay per-image distributions, split into comparison groups.

    ``kind="ecdf"`` is robust for long-tailed errors. ``kind="hist"`` draws
    density-normalized step histograms. Quantile clipping is shared by every
    label inside a group so FP/INT8 remain directly comparable.
    """
    if kind not in {"ecdf", "hist"}:
        raise ValueError("kind must be 'ecdf' or 'hist'")
    if clip_quantile is not None and not 0.0 < clip_quantile <= 1.0:
        raise ValueError("clip_quantile must be in (0, 1]")

    frames = _load_vs_logs(logs, labels=labels, groups=groups)
    fields = list(fields)
    unique_groups = list(
        dict.fromkeys(frame["_vs_group"].iloc[0] for frame in frames)
    )
    unique_labels = list(
        dict.fromkeys(frame["_vs_label"].iloc[0] for frame in frames)
    )
    palette = dict(
        zip(unique_labels, sns.color_palette(n_colors=len(unique_labels)))
    )
    fig, axs = plt.subplots(
        nrows=len(unique_groups),
        ncols=len(fields),
        figsize=(4.5 * len(fields), 3.6 * len(unique_groups)),
        squeeze=False,
    )

    for row_index, group in enumerate(unique_groups):
        group_frames = [
            frame for frame in frames
            if frame["_vs_group"].iloc[0] == group
        ]
        for column_index, field in enumerate(fields):
            ax = axs[row_index, column_index]
            value_sets = {}
            for frame in group_frames:
                if field not in frame:
                    raise KeyError(
                        f"missing field '{field}' in "
                        f"{frame['_vs_path'].iloc[0]}"
                    )
                label = frame["_vs_label"].iloc[0]
                value_sets[label] = pd.to_numeric(
                    frame[field], errors="coerce"
                ).dropna().to_numpy()

            nonempty = [values for values in value_sets.values() if len(values)]
            upper = None
            if nonempty and clip_quantile is not None:
                upper = float(
                    np.quantile(np.concatenate(nonempty), clip_quantile)
                )
            for label, values in value_sets.items():
                if upper is not None:
                    values = values[values <= upper]
                if not len(values):
                    continue
                if kind == "ecdf":
                    ordered = np.sort(values)
                    probability = (
                        np.arange(1, len(ordered) + 1) / len(ordered)
                    )
                    ax.plot(
                        ordered,
                        probability,
                        label=label,
                        color=palette[label],
                        linewidth=2,
                    )
                    ax.set_ylabel("CDF")
                else:
                    ax.hist(
                        values,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=2,
                        label=label,
                        color=palette[label],
                    )
                    ax.set_ylabel("Density")
            suffix = (
                f" (≤q{clip_quantile:.2f})"
                if clip_quantile is not None and clip_quantile < 1.0
                else ""
            )
            ax.set_title(f"{group}: {field}{suffix}")
            ax.grid(alpha=0.2)
            if row_index == 0 and column_index == 0:
                ax.legend()

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
