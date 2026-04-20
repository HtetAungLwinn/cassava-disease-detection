from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render artifact plots. "
            "Install it in the notebook environment before running this dashboard."
        ) from exc
    return plt


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "artifacts").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root. Run this notebook from somewhere inside the repository."
    )


def _candidate_artifact_dirs(repo_root: Path, model_family: str) -> list[Path]:
    family_token = _normalize_token(model_family)
    family_root = repo_root / "artifacts" / model_family

    search_roots: list[Path] = []
    if family_root.exists():
        search_roots.append(family_root)
    search_roots.append(repo_root)

    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        for summary_path in root.rglob("summary_metrics.json"):
            parent = summary_path.parent.resolve()
            normalized_parent = _normalize_token(str(parent))
            normalized_file = _normalize_token(str(summary_path))
            if family_token not in normalized_parent and family_token not in normalized_file:
                continue
            if parent in seen:
                continue
            candidates.append(parent)
            seen.add(parent)

    return sorted(
        candidates,
        key=lambda path: (path / "summary_metrics.json").stat().st_mtime,
        reverse=True,
    )


def resolve_artifact_dir(
    model_family: str,
    preferred_subdir: str | None = None,
    override: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or find_repo_root()

    if override:
        artifact_dir = Path(override).expanduser().resolve()
        summary_path = artifact_dir / "summary_metrics.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Override directory does not contain summary_metrics.json: {artifact_dir}"
            )
        return artifact_dir

    candidates = _candidate_artifact_dirs(repo_root, model_family)
    if not candidates:
        family_root = repo_root / "artifacts" / model_family
        raise FileNotFoundError(
            "No saved evaluation artifacts were found for "
            f"{model_family}. Looked under {family_root}. "
            "Rerun the final evaluation/save cells in the model notebook so it writes "
            "summary_metrics.json, classification_report.csv, training_history.csv, "
            "confusion_matrix_counts.csv, and validation_predictions.csv."
        )

    if preferred_subdir:
        preferred_token = _normalize_token(preferred_subdir)
        preferred_matches = [
            path for path in candidates if preferred_token in _normalize_token(str(path))
        ]
        if preferred_matches:
            return preferred_matches[0]
        warnings.warn(
            f"No artifact directory matched preferred_subdir={preferred_subdir!r}. "
            f"Falling back to the most recently modified {model_family} artifacts.",
            stacklevel=2,
        )

    return candidates[0]


def load_artifact_bundle(artifact_dir: str | Path) -> dict[str, object]:
    artifact_dir = Path(artifact_dir).resolve()

    def maybe_read_csv(filename: str, *, index_col: int | None = None) -> pd.DataFrame | None:
        file_path = artifact_dir / filename
        if not file_path.exists():
            return None
        return pd.read_csv(file_path, index_col=index_col)

    summary_path = artifact_dir / "summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_metrics.json in {artifact_dir}")

    return {
        "artifact_dir": artifact_dir,
        "summary_metrics": json.loads(summary_path.read_text(encoding="utf-8")),
        "history_df": maybe_read_csv("training_history.csv"),
        "report_df": maybe_read_csv("classification_report.csv", index_col=0),
        "conf_mat_df": maybe_read_csv("confusion_matrix_counts.csv", index_col=0),
        "valid_results_df": maybe_read_csv("validation_predictions.csv"),
    }


def _percent_value(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def build_summary_table(
    summary_metrics: dict[str, object], report_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    def fallback(metric_key: str, row_label: str, column_label: str) -> float | None:
        if metric_key in summary_metrics:
            return float(summary_metrics[metric_key])
        if (
            report_df is not None
            and row_label in report_df.index
            and column_label in report_df.columns
        ):
            return float(report_df.loc[row_label, column_label])
        return None

    records = [
        (
            "Validation Accuracy",
            fallback("validation_accuracy_from_predictions", "accuracy", "precision"),
        ),
        ("Macro Precision", fallback("macro_precision", "macro avg", "precision")),
        ("Macro Recall", fallback("macro_recall", "macro avg", "recall")),
        ("Macro F1", fallback("macro_f1", "macro avg", "f1-score")),
        ("Weighted F1", fallback("weighted_f1", "weighted avg", "f1-score")),
    ]

    summary_df = pd.DataFrame(records, columns=["Metric", "Raw Score"])
    summary_df["Score (%)"] = summary_df["Raw Score"].apply(_percent_value)
    return summary_df


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def plot_summary_metrics(summary_df: pd.DataFrame, model_family: str) -> None:
    plt = _get_plt()
    plot_df = summary_df.dropna(subset=["Score (%)"]).copy()
    if plot_df.empty:
        display(Markdown("No summary metrics available to plot."))
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(plot_df["Metric"], plot_df["Score (%)"], color="#4C78A8")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"{model_family} summary metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, plot_df["Score (%)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_training_history(history_df: pd.DataFrame, model_family: str) -> None:
    plt = _get_plt()
    required_accuracy_columns = {"epoch", "train_accuracy", "valid_accuracy"}
    required_loss_columns = {"epoch", "train_loss", "valid_loss"}

    if history_df is None or history_df.empty:
        display(Markdown("Training history was not found in the artifact directory."))
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if required_accuracy_columns.issubset(history_df.columns):
        axes[0].plot(
            history_df["epoch"],
            history_df["train_accuracy"],
            marker="o",
            linewidth=2,
            label="Train accuracy",
        )
        axes[0].plot(
            history_df["epoch"],
            history_df["valid_accuracy"],
            marker="o",
            linewidth=2,
            label="Validation accuracy",
        )
        axes[0].set_title(f"{model_family} accuracy over epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
    else:
        axes[0].set_visible(False)

    if required_loss_columns.issubset(history_df.columns):
        axes[1].plot(
            history_df["epoch"],
            history_df["train_loss"],
            marker="o",
            linewidth=2,
            label="Train loss",
        )
        axes[1].plot(
            history_df["epoch"],
            history_df["valid_loss"],
            marker="o",
            linewidth=2,
            label="Validation loss",
        )
        axes[1].set_title(f"{model_family} loss over epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_per_class_metrics(report_df: pd.DataFrame, model_family: str) -> pd.DataFrame | None:
    plt = _get_plt()
    if report_df is None or report_df.empty:
        display(Markdown("Classification report was not found in the artifact directory."))
        return None

    excluded_rows = {"accuracy", "macro avg", "weighted avg"}
    class_rows = [row for row in report_df.index if row not in excluded_rows]
    if not class_rows:
        display(Markdown("Classification report does not contain per-class rows."))
        return None

    per_class_df = report_df.loc[class_rows, ["precision", "recall", "f1-score", "support"]].copy()
    display(per_class_df.round(3))

    metrics_plot_df = per_class_df[["precision", "recall", "f1-score"]].mul(100.0)
    ax = metrics_plot_df.plot(kind="bar", figsize=(12, 5))
    ax.set_title(f"{model_family} per-class precision, recall, and F1")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    return per_class_df


def plot_confusion_matrix(conf_mat_df: pd.DataFrame, model_family: str) -> None:
    plt = _get_plt()
    if conf_mat_df is None or conf_mat_df.empty:
        display(Markdown("Confusion matrix counts were not found in the artifact directory."))
        return

    labels = list(conf_mat_df.columns)
    matrix = conf_mat_df.to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{model_family} validation confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


def display_top_mistakes(valid_results_df: pd.DataFrame) -> pd.DataFrame | None:
    if valid_results_df is None or valid_results_df.empty or "correct" not in valid_results_df.columns:
        display(Markdown("Validation prediction details were not found in the artifact directory."))
        return None

    mistakes_df = valid_results_df.loc[~valid_results_df["correct"].astype(bool)].copy()
    if mistakes_df.empty:
        display(Markdown("No incorrect validation predictions were recorded in this artifact set."))
        return mistakes_df

    sort_columns = ["confidence"] if "confidence" in mistakes_df.columns else []
    if sort_columns:
        mistakes_df = mistakes_df.sort_values(sort_columns, ascending=False)

    preferred_columns = [
        column
        for column in [
            "image_id",
            "true_name",
            "pred_name",
            "label",
            "pred_label",
            "confidence",
        ]
        if column in mistakes_df.columns
    ]
    display(mistakes_df[preferred_columns].head(10))
    return mistakes_df


def display_artifact_dashboard(
    model_family: str,
    preferred_subdir: str | None = None,
    override: str | Path | None = None,
) -> dict[str, object]:
    repo_root = find_repo_root()
    artifact_dir = resolve_artifact_dir(
        model_family=model_family,
        preferred_subdir=preferred_subdir,
        override=override,
        repo_root=repo_root,
    )
    bundle = load_artifact_bundle(artifact_dir)

    summary_metrics = bundle["summary_metrics"]
    report_df = bundle["report_df"]
    history_df = bundle["history_df"]
    conf_mat_df = bundle["conf_mat_df"]
    valid_results_df = bundle["valid_results_df"]

    display(Markdown(f"## {model_family} Artifact Dashboard"))
    display(
        Markdown(
            f"Using saved artifacts from `{_relative_path(artifact_dir, repo_root)}`"
        )
    )

    display(Markdown("### Summary Metrics"))
    summary_df = build_summary_table(summary_metrics, report_df)
    display(summary_df[["Metric", "Score (%)"]].round({"Score (%)": 2}))
    plot_summary_metrics(summary_df, model_family)

    display(Markdown("### Training Curves"))
    plot_training_history(history_df, model_family)

    display(Markdown("### Per-Class Metrics"))
    plot_per_class_metrics(report_df, model_family)

    display(Markdown("### Confusion Matrix"))
    plot_confusion_matrix(conf_mat_df, model_family)

    display(Markdown("### Highest-Confidence Validation Mistakes"))
    display_top_mistakes(valid_results_df)

    return bundle
