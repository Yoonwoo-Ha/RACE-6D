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

                is_higher_better = field in higher_is_better
                is_lower_better = not is_higher_better and (
                    field.startswith("loss_")
                    or field.startswith("metric_")
                    or field in lower_is_better_extra
                )

                train_field = f"train_{field}"
                if train_field in df_ewm.columns:
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

                test_field = f"test_{field}"
                if test_field in df_ewm.columns and not df_ewm[test_field].isna().all():
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

        if field in higher_is_better:
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

                is_higher_better = field in higher_is_better
                is_lower_better = not is_higher_better and (
                    field.startswith("loss_")
                    or field.startswith("metric_")
                    or field in lower_is_better_extra
                )
                train_field = f"train_{field}"
                if train_field in df_ewm.columns:
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

                test_field = f"test_{field}"
                if test_field in df_ewm.columns and not df_ewm[test_field].isna().all():
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

        if field in higher_is_better:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3)


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
