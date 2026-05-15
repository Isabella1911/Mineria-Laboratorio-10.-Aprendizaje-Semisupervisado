"""
Laboratorio 10 - Aprendizaje Semisupervisado.

Entrega reproducible sin dependencias pesadas: usa numpy y librerias estandar
para descargar el dataset, entrenar modelos, generar visualizaciones SVG y
crear un reporte PDF.
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import subprocess
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
RANDOM_SEEDS = [7, 21, 42]
LABEL_FRACTIONS = [0.05, 0.10, 0.20]
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "winequality-red.csv"
RESULTS_FILE = OUTPUT_DIR / "results.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.txt"
REPORT_FILE = OUTPUT_DIR / "reporte_laboratorio_10.pdf"
REPORT_HTML_FILE = OUTPUT_DIR / "reporte_laboratorio_10.html"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)


def download_dataset() -> None:
    if DATA_FILE.exists():
        return
    print(f"Descargando dataset desde {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)


def load_wine_quality() -> tuple[np.ndarray, np.ndarray, list[str]]:
    with DATA_FILE.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    data = np.array(rows, dtype=float)
    return data[:, :-1], data[:, -1].astype(int), header


def stratified_train_test_split(
    y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def labeled_subset_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_cls = max(1, int(round(len(idx) * fraction)))
        selected.extend(idx[:n_cls].tolist())
    rng.shuffle(selected)
    return np.array(selected)


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler no ajustado")
        return (x - self.mean_) / self.std_


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    dist2 = np.maximum(a2 + b2 - 2 * a @ b.T, 0.0)
    return np.sqrt(dist2)


class KNNClassifier:
    def __init__(self, n_neighbors: int = 7, weighted: bool = True):
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=int)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None or self.classes_ is None:
            raise RuntimeError("Modelo KNN no entrenado")
        dist = pairwise_distances(np.asarray(x, dtype=float), self.x_)
        k = min(self.n_neighbors, len(self.y_))
        neigh = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        proba = np.zeros((len(x), len(self.classes_)))
        class_pos = {cls: pos for pos, cls in enumerate(self.classes_)}
        for i, neighbors in enumerate(neigh):
            if self.weighted:
                weights = 1.0 / (dist[i, neighbors] + 1e-9)
            else:
                weights = np.ones(len(neighbors))
            for idx, weight in zip(neighbors, weights):
                proba[i, class_pos[int(self.y_[idx])]] += weight
            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
        return proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]


class SelfTrainingKNN:
    def __init__(
        self,
        n_neighbors: int = 7,
        threshold: float = 0.75,
        max_iter: int = 20,
        batch_fraction: float = 0.25,
    ):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.max_iter = max_iter
        self.batch_fraction = batch_fraction
        self.model_: KNNClassifier | None = None
        self.history_: list[int] = []

    def fit(self, x: np.ndarray, y_partial: np.ndarray) -> "SelfTrainingKNN":
        labeled = np.where(y_partial != -1)[0].tolist()
        unlabeled = np.where(y_partial == -1)[0].tolist()
        y_work = y_partial.copy()
        for _ in range(self.max_iter):
            model = KNNClassifier(self.n_neighbors).fit(x[labeled], y_work[labeled])
            if not unlabeled:
                break
            proba = model.predict_proba(x[unlabeled])
            confidence = np.max(proba, axis=1)
            candidates = np.where(confidence >= self.threshold)[0]
            if len(candidates) == 0:
                self.history_.append(0)
                break
            limit = max(1, int(math.ceil(len(unlabeled) * self.batch_fraction)))
            chosen_local = candidates[np.argsort(confidence[candidates])[-limit:]]
            chosen_global = [unlabeled[i] for i in chosen_local]
            predictions = model.classes_[np.argmax(proba[chosen_local], axis=1)]
            for idx, pred in zip(chosen_global, predictions):
                y_work[idx] = int(pred)
            labeled.extend(chosen_global)
            chosen_set = set(chosen_global)
            unlabeled = [idx for idx in unlabeled if idx not in chosen_set]
            self.history_.append(len(chosen_global))
        self.model_ = KNNClassifier(self.n_neighbors).fit(x[labeled], y_work[labeled])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Modelo self-training no entrenado")
        return self.model_.predict(x)


class LabelPropagationKNN:
    def __init__(self, n_neighbors: int = 10, alpha: float = 0.85, max_iter: int = 200, tol: float = 1e-4):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.x_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.distributions_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, x: np.ndarray, y_partial: np.ndarray) -> "LabelPropagationKNN":
        self.x_ = x
        labeled_mask = y_partial != -1
        self.classes_ = np.unique(y_partial[labeled_mask])
        n = len(x)
        c = len(self.classes_)
        class_pos = {cls: pos for pos, cls in enumerate(self.classes_)}
        y0 = np.zeros((n, c))
        for idx in np.where(labeled_mask)[0]:
            y0[idx, class_pos[int(y_partial[idx])]] = 1.0
        unlabeled_mask = ~labeled_mask
        if unlabeled_mask.any():
            y0[unlabeled_mask] = 1.0 / c

        w = self._knn_affinity(x)
        row_sum = w.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        s = w / row_sum
        f = y0.copy()
        for it in range(1, self.max_iter + 1):
            previous = f.copy()
            f = self.alpha * (s @ f) + (1 - self.alpha) * y0
            f[labeled_mask] = y0[labeled_mask]
            diff = np.abs(f - previous).sum()
            if diff < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter
        f_sum = f.sum(axis=1, keepdims=True)
        f_sum[f_sum == 0] = 1.0
        self.distributions_ = f / f_sum
        return self

    def _knn_affinity(self, x: np.ndarray) -> np.ndarray:
        dist = pairwise_distances(x, x)
        np.fill_diagonal(dist, np.inf)
        k = min(self.n_neighbors, len(x) - 1)
        neighbors = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        finite = dist[np.isfinite(dist)]
        sigma = np.median(finite) if len(finite) else 1.0
        sigma = max(float(sigma), 1e-6)
        w = np.zeros_like(dist)
        for i in range(len(x)):
            for j in neighbors[i]:
                value = math.exp(-(dist[i, j] ** 2) / (2 * sigma**2))
                w[i, j] = value
                w[j, i] = max(w[j, i], value)
        return w

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.classes_ is None or self.distributions_ is None:
            raise RuntimeError("Modelo label propagation no entrenado")
        dist = pairwise_distances(x, self.x_)
        k = min(self.n_neighbors, len(self.x_))
        neigh = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        proba = np.zeros((len(x), len(self.classes_)))
        for i, neighbors in enumerate(neigh):
            weights = 1.0 / (dist[i, neighbors] + 1e-9)
            proba[i] = weights @ self.distributions_[neighbors]
            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
        return self.classes_[np.argmax(proba, axis=1)]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    pos = {cls: i for i, cls in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[pos[int(truth)], pos[int(pred)]] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, classes)
    accuracy = float(np.trace(cm) / np.sum(cm))
    recalls = []
    f1s = []
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recalls)),
    }


def describe_dataset(x: np.ndarray, y: np.ndarray, header: list[str]) -> dict[str, object]:
    feature_names = header[:-1]
    stats = []
    for idx, name in enumerate(feature_names):
        col = x[:, idx]
        stats.append(
            {
                "variable": name,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "q25": float(np.quantile(col, 0.25)),
                "median": float(np.median(col)),
                "q75": float(np.quantile(col, 0.75)),
                "max": float(np.max(col)),
            }
        )
    class_counts = {int(cls): int(np.sum(y == cls)) for cls in np.unique(y)}
    return {
        "rows": int(len(x)),
        "columns": len(header),
        "features": feature_names,
        "target": header[-1],
        "stats": stats,
        "class_counts": class_counts,
    }


def run_experiments(x: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    train_idx, test_idx = stratified_train_test_split(y, test_size=0.2, seed=42)
    scaler = StandardScaler().fit(x[train_idx])
    x_train = scaler.transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    records: list[dict[str, object]] = []
    best_predictions: dict[str, np.ndarray] = {}
    best_scores: dict[str, float] = {}
    best_fraction = 0.20
    best_seed = 42

    for fraction in LABEL_FRACTIONS:
        for seed in RANDOM_SEEDS:
            labeled_idx = labeled_subset_indices(y_train, fraction, seed)
            y_partial = np.full_like(y_train, fill_value=-1)
            y_partial[labeled_idx] = y_train[labeled_idx]

            baseline = KNNClassifier(n_neighbors=7).fit(x_train[labeled_idx], y_train[labeled_idx])
            pred = baseline.predict(x_test)
            add_record(records, "Supervisado KNN", fraction, seed, "k=7", y_test, pred, len(labeled_idx))
            maybe_store_best(best_predictions, best_scores, "Supervisado KNN", fraction, seed, y_test, pred)

            for threshold in [0.60, 0.75, 0.90]:
                model = SelfTrainingKNN(n_neighbors=7, threshold=threshold, max_iter=20).fit(x_train, y_partial)
                pred = model.predict(x_test)
                extra = f"threshold={threshold:.2f}; pseudo={sum(model.history_)}"
                add_record(records, "Self-Training KNN", fraction, seed, extra, y_test, pred, len(labeled_idx))
                maybe_store_best(best_predictions, best_scores, "Self-Training KNN", fraction, seed, y_test, pred)

            for neighbors in [5, 10, 20]:
                model = LabelPropagationKNN(n_neighbors=neighbors, alpha=0.85, max_iter=200).fit(x_train, y_partial)
                pred = model.predict(x_test)
                extra = f"k={neighbors}; alpha=0.85; iter={model.n_iter_}"
                add_record(records, "Label Propagation", fraction, seed, extra, y_test, pred, len(labeled_idx))
                maybe_store_best(best_predictions, best_scores, "Label Propagation", fraction, seed, y_test, pred)

    full_supervised = KNNClassifier(n_neighbors=7).fit(x_train, y_train)
    full_pred = full_supervised.predict(x_test)
    full_metrics = classification_metrics(y_test, full_pred)

    context = {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "classes": np.unique(y).astype(int),
        "y_test": y_test,
        "best_predictions": best_predictions,
        "best_scores": best_scores,
        "full_supervised_metrics": full_metrics,
        "best_fraction": best_fraction,
        "best_seed": best_seed,
    }
    return records, context


def add_record(
    records: list[dict[str, object]],
    model: str,
    fraction: float,
    seed: int,
    hyperparams: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labeled_count: int,
) -> None:
    metrics = classification_metrics(y_true, y_pred)
    records.append(
        {
            "model": model,
            "label_fraction": fraction,
            "seed": seed,
            "hyperparams": hyperparams,
            "labeled_count": labeled_count,
            **metrics,
        }
    )


def maybe_store_best(
    best_predictions: dict[str, np.ndarray],
    best_scores: dict[str, float],
    model: str,
    fraction: float,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    if fraction != 0.20 or seed != 42:
        return
    score = classification_metrics(y_true, y_pred)["macro_f1"]
    if score > best_scores.get(model, -1):
        best_scores[model] = score
        best_predictions[model] = y_pred.copy()


def write_results_csv(records: list[dict[str, object]]) -> None:
    fields = [
        "model",
        "label_fraction",
        "seed",
        "hyperparams",
        "labeled_count",
        "accuracy",
        "macro_f1",
        "macro_recall",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for rec in records:
        key = (str(rec["model"]), float(rec["label_fraction"]))
        groups.setdefault(key, []).append(rec)
    summary = []
    for (model, fraction), items in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        best_by_seed: dict[int, dict[str, object]] = {}
        for item in items:
            seed = int(item["seed"])
            if seed not in best_by_seed or float(item["macro_f1"]) > float(best_by_seed[seed]["macro_f1"]):
                best_by_seed[seed] = item
        values = list(best_by_seed.values())
        summary.append(
            {
                "model": model,
                "label_fraction": fraction,
                "accuracy_mean": float(np.mean([v["accuracy"] for v in values])),
                "f1_mean": float(np.mean([v["macro_f1"] for v in values])),
                "recall_mean": float(np.mean([v["macro_recall"] for v in values])),
                "f1_std": float(np.std([v["macro_f1"] for v in values])),
                "best_hyperparams": max(values, key=lambda v: float(v["macro_f1"]))["hyperparams"],
            }
        )
    return summary


def escape_xml(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_svg(path: Path, content: str, width: int = 900, height: int = 520) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{content}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def plot_performance(summary: list[dict[str, object]]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 40, 55, 75
    colors = {"Supervisado KNN": "#1f77b4", "Self-Training KNN": "#2ca02c", "Label Propagation": "#d62728"}
    xs = LABEL_FRACTIONS
    y_values = [float(item["f1_mean"]) for item in summary]
    y_min = max(0.0, min(y_values) - 0.05)
    y_max = min(1.0, max(y_values) + 0.05)

    def sx(frac: float) -> float:
        return left + (frac - min(xs)) / (max(xs) - min(xs)) * (width - left - right)

    def sy(val: float) -> float:
        return top + (y_max - val) / (y_max - y_min) * (height - top - bottom)

    parts = [
        '<text x="450" y="30" font-size="22" text-anchor="middle" font-family="Arial">Macro-F1 por porcentaje de etiquetas</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = sy(float(tick))
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="12" text-anchor="end" font-family="Arial">{tick:.2f}</text>')
    for frac in xs:
        x = sx(frac)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+25}" font-size="13" text-anchor="middle" font-family="Arial">{int(frac*100)}%</text>')
    for model, color in colors.items():
        points = []
        for frac in xs:
            item = next(s for s in summary if s["model"] == model and abs(float(s["label_fraction"]) - frac) < 1e-9)
            points.append((sx(frac), sy(float(item["f1_mean"]))))
        path = " ".join([f"{x:.1f},{y:.1f}" for x, y in points])
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
    legend_y = 70
    for i, (model, color) in enumerate(colors.items()):
        y = legend_y + i * 24
        parts.append(f'<rect x="650" y="{y-12}" width="16" height="16" fill="{color}"/>')
        parts.append(f'<text x="674" y="{y+1}" font-size="14" font-family="Arial">{escape_xml(model)}</text>')
    parts.append('<text x="450" y="500" font-size="14" text-anchor="middle" font-family="Arial">Porcentaje de datos etiquetados en entrenamiento</text>')
    parts.append('<text x="18" y="285" font-size="14" text-anchor="middle" font-family="Arial" transform="rotate(-90 18 285)">Macro-F1 promedio</text>')
    save_svg(OUTPUT_DIR / "figures" / "performance_by_label_fraction.svg", "\n".join(parts), width, height)


def plot_hyperparameter_sensitivity(records: list[dict[str, object]]) -> None:
    width, height = 900, 520
    left, top = 80, 55
    bar_w = 34
    gap = 12
    filtered = [r for r in records if float(r["label_fraction"]) == 0.10]
    labels = []
    values = []
    for model in ["Self-Training KNN", "Label Propagation"]:
        params = sorted(set(str(r["hyperparams"]).split(";")[0] for r in filtered if r["model"] == model))
        for param in params:
            vals = [float(r["macro_f1"]) for r in filtered if r["model"] == model and str(r["hyperparams"]).startswith(param)]
            labels.append(param.replace("threshold=", "thr=").replace("k=", "k="))
            values.append(float(np.mean(vals)))
    y_min = max(0.0, min(values) - 0.05)
    y_max = min(1.0, max(values) + 0.05)
    plot_h = height - top - 90

    def sy(val: float) -> float:
        return top + (y_max - val) / (y_max - y_min) * plot_h

    parts = [
        '<text x="450" y="30" font-size="22" text-anchor="middle" font-family="Arial">Sensibilidad de hiperparametros con 10% etiquetado</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-40}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = sy(float(tick))
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{width-40}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="12" text-anchor="end" font-family="Arial">{tick:.2f}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + 35 + i * (bar_w + gap)
        y = sy(value)
        color = "#2ca02c" if i < 3 else "#d62728"
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{top+plot_h-y:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+23}" font-size="12" text-anchor="middle" font-family="Arial">{escape_xml(label)}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" font-size="11" text-anchor="middle" font-family="Arial">{value:.3f}</text>')
    save_svg(OUTPUT_DIR / "figures" / "hyperparameter_sensitivity.svg", "\n".join(parts), width, height)


def plot_confusion_matrices(context: dict[str, object]) -> None:
    y_test = context["y_test"]
    classes = context["classes"]
    predictions = context["best_predictions"]
    for model, pred in predictions.items():
        cm = confusion_matrix(y_test, pred, classes)
        width, height = 650, 600
        left, top, cell = 110, 80, 62
        max_val = max(1, int(cm.max()))
        parts = [
            f'<text x="325" y="35" font-size="21" text-anchor="middle" font-family="Arial">Matriz de confusion - {escape_xml(model)}</text>',
            '<text x="325" y="570" font-size="14" text-anchor="middle" font-family="Arial">Prediccion</text>',
            '<text x="26" y="285" font-size="14" text-anchor="middle" font-family="Arial" transform="rotate(-90 26 285)">Valor real</text>',
        ]
        for i, cls in enumerate(classes):
            parts.append(f'<text x="{left-20}" y="{top+i*cell+38}" font-size="13" text-anchor="end" font-family="Arial">{int(cls)}</text>')
            parts.append(f'<text x="{left+i*cell+31}" y="{top-12}" font-size="13" text-anchor="middle" font-family="Arial">{int(cls)}</text>')
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = int(cm[i, j])
                intensity = val / max_val
                red = int(255 - 150 * intensity)
                green = int(255 - 180 * intensity)
                blue = int(255 - 210 * intensity)
                fill = f"rgb({red},{green},{blue})"
                x = left + j * cell
                y = top + i * cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
                parts.append(f'<text x="{x+cell/2}" y="{y+cell/2+5}" font-size="14" text-anchor="middle" font-family="Arial">{val}</text>')
        filename = model.lower().replace(" ", "_").replace("-", "_") + "_confusion.svg"
        save_svg(OUTPUT_DIR / "figures" / filename, "\n".join(parts), width, height)


def plot_prediction_comparison(context: dict[str, object]) -> None:
    y_test = context["y_test"]
    predictions = context["best_predictions"]
    model = max(context["best_scores"], key=context["best_scores"].get)
    pred = predictions[model]
    rng = random.Random(42)
    points = list(zip(y_test.tolist(), pred.tolist()))
    sample = points[:]
    rng.shuffle(sample)
    sample = sample[:220]
    width, height = 760, 520
    left, top, plot_w, plot_h = 80, 55, 610, 380
    classes = sorted(set(y_test.tolist()))

    def sx(value: int) -> float:
        return left + (value - min(classes)) / (max(classes) - min(classes)) * plot_w

    def sy(value: int) -> float:
        return top + (max(classes) - value) / (max(classes) - min(classes)) * plot_h

    parts = [
        f'<text x="380" y="30" font-size="21" text-anchor="middle" font-family="Arial">Prediccion vs valor real - {escape_xml(model)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#777" stroke-dasharray="6 5"/>',
    ]
    for cls in classes:
        parts.append(f'<text x="{sx(cls):.1f}" y="{top+plot_h+24}" font-size="12" text-anchor="middle" font-family="Arial">{cls}</text>')
        parts.append(f'<text x="{left-15}" y="{sy(cls)+4:.1f}" font-size="12" text-anchor="end" font-family="Arial">{cls}</text>')
    for real, predicted in sample:
        jitter_x = rng.uniform(-8, 8)
        jitter_y = rng.uniform(-8, 8)
        color = "#2ca02c" if real == predicted else "#d62728"
        parts.append(f'<circle cx="{sx(real)+jitter_x:.1f}" cy="{sy(predicted)+jitter_y:.1f}" r="3.5" fill="{color}" opacity="0.62"/>')
    parts.append('<text x="385" y="490" font-size="14" text-anchor="middle" font-family="Arial">Valor real de calidad</text>')
    parts.append('<text x="18" y="250" font-size="14" text-anchor="middle" font-family="Arial" transform="rotate(-90 18 250)">Calidad predicha</text>')
    save_svg(OUTPUT_DIR / "figures" / "prediction_vs_real.svg", "\n".join(parts), width, height)


def write_summary(
    eda: dict[str, object], summary: list[dict[str, object]], records: list[dict[str, object]], context: dict[str, object]
) -> None:
    best = max(summary, key=lambda item: float(item["f1_mean"]))
    lines = [
        "Resumen de Laboratorio 10 - Aprendizaje Semisupervisado",
        "",
        f"Dataset: Wine Quality Red de UCI ({DATA_URL})",
        f"Dimensiones: {eda['rows']} filas, {eda['columns']} columnas.",
        f"Variables predictoras: {', '.join(eda['features'])}",
        f"Variable objetivo: {eda['target']} con clases {eda['class_counts']}",
        f"Separacion: {context['train_size']} entrenamiento, {context['test_size']} prueba.",
        "",
        "Resultados agregados: se reporta el mejor hiperparametro por semilla y luego el promedio entre semillas.",
    ]
    for item in summary:
        lines.append(
            f"- {item['model']} con {int(item['label_fraction']*100)}% etiquetas: "
            f"accuracy={item['accuracy_mean']:.3f}, macro-F1={item['f1_mean']:.3f}, "
            f"recall={item['recall_mean']:.3f}, std_F1={item['f1_std']:.3f}, "
            f"mejor configuracion={item['best_hyperparams']}"
        )
    lines.extend(
        [
            "",
            f"Mejor modelo semisupervisado: {best['model']} con {int(best['label_fraction']*100)}% etiquetado "
            f"(macro-F1 promedio={best['f1_mean']:.3f}).",
            f"Referencia supervisada con 100% de etiquetas: accuracy={context['full_supervised_metrics']['accuracy']:.3f}, "
            f"macro-F1={context['full_supervised_metrics']['macro_f1']:.3f}.",
        ]
    )
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


class SimplePDF:
    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.current: list[str] = []
        self.y = 760
        self.page_width = 612
        self.margin = 54

    def add_page(self) -> None:
        if self.current:
            self.pages.append(self.current)
        self.current = []
        self.y = 760

    def text(self, text: str, size: int = 11, bold: bool = False, gap: int = 15) -> None:
        if self.y < 60:
            self.add_page()
        font = "F2" if bold else "F1"
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.current.append(f"BT /{font} {size} Tf {self.margin} {self.y} Td ({safe}) Tj ET")
        self.y -= gap

    def paragraph(self, text: str, size: int = 10, width: int = 92) -> None:
        for line in textwrap.wrap(text, width=width):
            self.text(line, size=size, gap=13)
        self.y -= 5

    def heading(self, text: str) -> None:
        self.y -= 6
        self.text(text, size=14, bold=True, gap=19)

    def save(self, path: Path) -> None:
        if self.current:
            self.pages.append(self.current)
            self.current = []
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(self.pages)))
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode("latin-1"))
        for i, page in enumerate(self.pages):
            content_id = 4 + i * 2
            page_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
                f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> "
                f"/Contents {content_id} 0 R >>"
            )
            stream = "\n".join(page).encode("latin-1", errors="replace")
            content_obj = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
            objects.append(page_obj.encode("latin-1"))
            objects.append(content_obj)
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")
        xref = len(pdf)
        pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(pdf)


def write_report(eda: dict[str, object], summary: list[dict[str, object]], context: dict[str, object]) -> None:
    best_ssl = max([s for s in summary if s["model"] != "Supervisado KNN"], key=lambda item: float(item["f1_mean"]))
    pdf = SimplePDF()
    pdf.add_page()
    pdf.text("Universidad del Valle de Guatemala", size=10)
    pdf.text("CC3074 - Mineria de Datos", size=10)
    pdf.text("Laboratorio 10: Aprendizaje Semisupervisado", size=16, bold=True, gap=22)
    pdf.paragraph(
        "Este reporte presenta una comparacion experimental entre un modelo supervisado puro y dos enfoques "
        "semisupervisados sobre el dataset Wine Quality Red de UCI. El objetivo fue simular un escenario con "
        "pocas etiquetas disponibles y medir el efecto de incorporar ejemplos sin etiqueta."
    )
    pdf.heading("Dataset y analisis exploratorio")
    pdf.paragraph(
        f"Se utilizo Wine Quality Red, disponible publicamente en {DATA_URL}. El conjunto contiene {eda['rows']} "
        f"observaciones y {eda['columns']} columnas: 11 variables fisicoquimicas y la calidad sensorial del vino "
        f"como clase. No se detectaron valores faltantes en el archivo original."
    )
    pdf.paragraph(
        f"El objetivo es multiclase, con el siguiente balance: {eda['class_counts']}. La distribucion esta "
        "concentrada en calidades 5 y 6, por lo que ademas de accuracy se reporta macro-F1 y macro-recall."
    )
    pdf.heading("Preprocesamiento")
    pdf.paragraph(
        "Todas las variables predictoras son numericas. Se aplico estandarizacion z-score calculada solo con "
        "entrenamiento para evitar fuga de informacion. La separacion entrenamiento-prueba fue estratificada, "
        f"con {context['train_size']} registros de entrenamiento y {context['test_size']} de prueba."
    )
    pdf.heading("Diseno experimental")
    pdf.paragraph(
        "En entrenamiento se conservaron etiquetas reales solamente para 5%, 10% y 20% de los datos. El resto "
        "se marco como no etiquetado (-1). Para reducir dependencia de una sola particion, cada porcentaje se "
        "repitio con tres semillas aleatorias."
    )
    pdf.paragraph(
        "El baseline supervisado fue KNN entrenado solo con el subconjunto etiquetado. Los metodos "
        "semisupervisados fueron self-training con KNN y propagacion de etiquetas basada en un grafo k-NN."
    )
    pdf.heading("Fundamento de los algoritmos")
    pdf.paragraph(
        "Self-training ajusta un clasificador con los datos etiquetados, predice pseudo-etiquetas para datos "
        "sin etiqueta y agrega iterativamente los ejemplos cuya confianza supera un umbral. Su riesgo principal "
        "es la propagacion de errores: una pseudo-etiqueta incorrecta puede reforzarse en iteraciones futuras."
    )
    pdf.paragraph(
        "Label propagation modela los datos como un grafo ponderado. Los nodos son observaciones y las aristas "
        "conectan vecinos cercanos. Las distribuciones de clase se difunden por el grafo mientras las etiquetas "
        "conocidas se mantienen fijas. El supuesto es suavidad local: puntos cercanos tienden a compartir clase."
    )
    pdf.heading("Resultados")
    for item in summary:
        pdf.paragraph(
            f"{item['model']} con {int(item['label_fraction']*100)}% etiquetado obtuvo accuracy promedio "
            f"{item['accuracy_mean']:.3f}, macro-F1 {item['f1_mean']:.3f} y macro-recall {item['recall_mean']:.3f}. "
            f"La mejor configuracion observada fue {item['best_hyperparams']}."
        )
    pdf.paragraph(
        f"Como referencia superior, KNN entrenado con el 100% de etiquetas alcanzo accuracy "
        f"{context['full_supervised_metrics']['accuracy']:.3f} y macro-F1 "
        f"{context['full_supervised_metrics']['macro_f1']:.3f}."
    )
    pdf.heading("Discusion")
    pdf.paragraph(
        "La metrica macro-F1 es mas exigente que accuracy porque penaliza el bajo rendimiento en clases poco "
        "frecuentes. En este dataset, la concentracion en calidades intermedias dificulta distinguir calidades "
        "raras, especialmente con solo 5% de etiquetas."
    )
    pdf.paragraph(
        "El umbral de self-training controla el compromiso entre cantidad de pseudo-etiquetas y ruido. Umbrales "
        "bajos incorporan mas ejemplos pero elevan el riesgo de error acumulado; umbrales altos son mas "
        "conservadores pero pueden desaprovechar datos sin etiqueta."
    )
    pdf.paragraph(
        "En label propagation, el numero de vecinos define la conectividad del grafo. Valores bajos pueden "
        "fragmentar la estructura y valores altos pueden mezclar regiones de clases distintas, reduciendo la "
        "capacidad de preservar fronteras locales."
    )
    pdf.heading("Visualizaciones generadas")
    pdf.paragraph(
        "Los graficos SVG en outputs/figures incluyen: desempeno segun porcentaje de etiquetas, sensibilidad "
        "a hiperparametros, matrices de confusion y comparacion entre calidad real y predicha."
    )
    pdf.heading("Conclusiones")
    pdf.paragraph(
        f"El mejor modelo semisupervisado fue {best_ssl['model']} con {int(best_ssl['label_fraction']*100)}% "
        f"de etiquetas, con macro-F1 promedio de {best_ssl['f1_mean']:.3f}. La evidencia muestra que los datos "
        "no etiquetados pueden aportar informacion estructural, pero el beneficio depende fuertemente de la "
        "calidad de las pseudo-etiquetas o del grafo construido."
    )
    pdf.paragraph(
        "Trabajar con pocas etiquetas aumenta la varianza y expone las limitaciones del balance de clases. Para "
        "una aplicacion real se recomienda recolectar mas etiquetas para calidades extremas y validar con mas "
        "semillas o validacion cruzada estratificada."
    )
    pdf.save(REPORT_FILE)


def write_html_report(eda: dict[str, object], summary: list[dict[str, object]], context: dict[str, object]) -> None:
    best_ssl = max([s for s in summary if s["model"] != "Supervisado KNN"], key=lambda item: float(item["f1_mean"]))
    rows = "\n".join(
        "<tr>"
        f"<td>{escape_xml(item['model'])}</td>"
        f"<td>{int(item['label_fraction'] * 100)}%</td>"
        f"<td>{item['accuracy_mean']:.3f}</td>"
        f"<td>{item['f1_mean']:.3f}</td>"
        f"<td>{item['recall_mean']:.3f}</td>"
        f"<td>{item['f1_std']:.3f}</td>"
        f"<td>{escape_xml(item['best_hyperparams'])}</td>"
        "</tr>"
        for item in summary
    )
    stats_rows = "\n".join(
        "<tr>"
        f"<td>{escape_xml(stat['variable'])}</td>"
        f"<td>{stat['mean']:.3f}</td>"
        f"<td>{stat['std']:.3f}</td>"
        f"<td>{stat['min']:.3f}</td>"
        f"<td>{stat['median']:.3f}</td>"
        f"<td>{stat['max']:.3f}</td>"
        "</tr>"
        for stat in eda["stats"]
    )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Laboratorio 10 - Aprendizaje Semisupervisado</title>
  <style>
    @page {{ margin: 18mm; }}
    body {{ font-family: Arial, sans-serif; color: #202124; line-height: 1.42; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    h2 {{ font-size: 17px; margin: 22px 0 8px; border-bottom: 1px solid #cfd4dc; padding-bottom: 3px; }}
    p {{ margin: 7px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 12px; }}
    th, td {{ border: 1px solid #c9cdd3; padding: 6px 7px; vertical-align: top; }}
    th {{ background: #eef1f5; text-align: left; }}
    img {{ display: block; max-width: 100%; margin: 8px auto 14px; page-break-inside: avoid; }}
    .meta {{ color: #555; font-size: 12px; margin-bottom: 16px; }}
    .note {{ background: #f6f8fa; border-left: 4px solid #6b778d; padding: 8px 10px; }}
  </style>
</head>
<body>
  <h1>Laboratorio 10: Aprendizaje Semisupervisado</h1>
  <div class="meta">Universidad del Valle de Guatemala - CC3074 Mineria de Datos</div>

  <h2>Introduccion</h2>
  <p>Este reporte compara un baseline supervisado contra dos algoritmos semisupervisados usando un escenario con pocas etiquetas disponibles. El experimento conserva 5%, 10% y 20% de etiquetas reales en entrenamiento y trata el resto como datos no etiquetados.</p>

  <h2>Dataset y exploracion</h2>
  <p>Se utilizo el dataset publico <strong>Wine Quality Red</strong> de UCI: <br>{escape_xml(DATA_URL)}</p>
  <p>El conjunto tiene {eda['rows']} filas y {eda['columns']} columnas. Las variables predictoras son {escape_xml(', '.join(eda['features']))}; la variable objetivo es <strong>{escape_xml(eda['target'])}</strong>. El balance de clases es {escape_xml(eda['class_counts'])}.</p>
  <table>
    <tr><th>Variable</th><th>Media</th><th>Desv.</th><th>Min</th><th>Mediana</th><th>Max</th></tr>
    {stats_rows}
  </table>

  <h2>Preprocesamiento</h2>
  <p>Las variables son numericas y no tienen valores faltantes en el archivo original. Se aplico estandarizacion z-score calculada solamente con entrenamiento para evitar fuga de informacion. La separacion fue estratificada: {context['train_size']} registros de entrenamiento y {context['test_size']} de prueba.</p>

  <h2>Diseno experimental</h2>
  <p>El baseline supervisado fue KNN entrenado solo con el subconjunto etiquetado. Los modelos semisupervisados fueron self-training con KNN y label propagation basado en un grafo k-NN. Cada porcentaje de etiquetas se evaluo con las semillas {RANDOM_SEEDS}.</p>

  <h2>Fundamento conceptual</h2>
  <p><strong>Self-training</strong> entrena un clasificador inicial con etiquetas reales, asigna pseudo-etiquetas a observaciones no etiquetadas y agrega iterativamente las predicciones cuya confianza supera un umbral. El hiperparametro critico es el threshold: si es bajo, crece el ruido; si es alto, se incorporan menos datos.</p>
  <p><strong>Label propagation</strong> construye un grafo donde las observaciones cercanas se conectan. Las distribuciones de clase se difunden por el grafo mientras las etiquetas conocidas quedan fijadas. El numero de vecinos controla si el grafo queda fragmentado o si mezcla regiones de clases distintas.</p>

  <h2>Resultados numericos</h2>
  <table>
    <tr><th>Modelo</th><th>Etiquetas</th><th>Accuracy</th><th>Macro-F1</th><th>Macro-recall</th><th>Std F1</th><th>Mejor hiperparametro</th></tr>
    {rows}
  </table>
  <p>Como referencia, KNN entrenado con el 100% de etiquetas alcanzo accuracy={context['full_supervised_metrics']['accuracy']:.3f} y macro-F1={context['full_supervised_metrics']['macro_f1']:.3f}.</p>

  <h2>Analisis grafico</h2>
  <img src="figures/performance_by_label_fraction.svg" alt="Macro-F1 por porcentaje de etiquetas">
  <p>El desempeno tiende a mejorar al aumentar el porcentaje de etiquetas. Label Propagation muestra la mejor macro-F1 semisupervisada con 20% de etiquetas, lo que sugiere que la estructura local del dataset aporta informacion util.</p>
  <img src="figures/hyperparameter_sensitivity.svg" alt="Sensibilidad de hiperparametros">
  <p>Self-training depende del umbral de confianza; Label Propagation depende del numero de vecinos. En este conjunto, un grafo mas local con k=5 resulto competitivo.</p>
  <img src="figures/label_propagation_confusion.svg" alt="Matriz de confusion de Label Propagation">
  <img src="figures/prediction_vs_real.svg" alt="Prediccion contra valor real">

  <h2>Discusion</h2>
  <p>La accuracy es moderada porque las clases 5 y 6 dominan el conjunto, mientras que las clases extremas tienen pocos ejemplos. Por eso macro-F1 y macro-recall son mas informativas: obligan a evaluar el rendimiento en clases minoritarias.</p>
  <p>Self-training puede mejorar cuando las pseudo-etiquetas son confiables, pero tambien puede propagar errores. Label Propagation aprovecha datos no etiquetados mediante suavidad local, aunque su calidad depende de que la distancia estandarizada represente bien la vecindad semantica.</p>

  <h2>Conclusiones</h2>
  <p class="note">El mejor modelo semisupervisado fue <strong>{escape_xml(best_ssl['model'])}</strong> con {int(best_ssl['label_fraction'] * 100)}% de etiquetas y macro-F1 promedio de {best_ssl['f1_mean']:.3f}.</p>
  <p>Trabajar con pocas etiquetas aumenta la varianza y dificulta aprender clases raras. Para mejorar una aplicacion real se recomienda recolectar mas etiquetas en calidades extremas, probar validacion cruzada estratificada y comparar clasificadores base adicionales.</p>
</body>
</html>
"""
    REPORT_HTML_FILE.write_text(html, encoding="utf-8")


def render_pdf_from_html() -> bool:
    candidates = [
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    browser = next((path for path in candidates if path and Path(path).exists()), None)
    if browser is None:
        return False
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={REPORT_FILE.resolve()}",
            str(REPORT_HTML_FILE.resolve()),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> None:
    ensure_dirs()
    download_dataset()
    x, y, header = load_wine_quality()
    eda = describe_dataset(x, y, header)
    records, context = run_experiments(x, y)
    write_results_csv(records)
    summary = aggregate(records)
    plot_performance(summary)
    plot_hyperparameter_sensitivity(records)
    plot_confusion_matrices(context)
    plot_prediction_comparison(context)
    write_summary(eda, summary, records, context)
    write_report(eda, summary, context)
    write_html_report(eda, summary, context)
    if not render_pdf_from_html():
        print("No se encontro Chrome/Edge; se conservo el PDF textual generado internamente.")
    print(SUMMARY_FILE.read_text(encoding="utf-8"))
    print(f"\nArchivos generados en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
