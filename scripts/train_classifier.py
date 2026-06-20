from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultrasound_nlp.nlp_pipeline import MODELS_DIR, ORGAN_CONFIG, records_for_organ, tokenize


def _records_by_split(organ: str):
    records = records_for_organ(organ)
    return {
        "train": [record for record in records if record.split == "train"],
        "test": [record for record in records if record.split == "test"],
        "val": [record for record in records if record.split == "val"],
    }


def _make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    max_features=8000,
                    min_df=2,
                    ngram_range=(1, 2),
                    lowercase=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )


def _evaluate(pipeline: Pipeline, records) -> dict:
    if not records:
        return {
            "count": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "labels": [],
            "confusion_matrix": [],
            "classification_report": {},
        }

    x = [record.finding for record in records]
    y_true = [record.label for record in records]
    y_pred = pipeline.predict(x)
    labels = sorted(set(y_true) | set(int(value) for value in y_pred))
    return {
        "count": len(records),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def train_one_organ(organ: str) -> dict:
    splits = _records_by_split(organ)
    train_records = splits["train"]
    if not train_records:
        raise ValueError(f"No training records found for {organ}")

    pipeline = _make_pipeline()
    x_train = [record.finding for record in train_records]
    y_train = [record.label for record in train_records]
    pipeline.fit(x_train, y_train)

    model_path = MODELS_DIR / f"{organ}_classifier.pkl"
    bundle = {
        "organ": organ,
        "organ_name": ORGAN_CONFIG[organ]["name"],
        "model_name": "TF-IDF + LogisticRegression",
        "pipeline": pipeline,
        "train_label_distribution": dict(Counter(y_train)),
    }
    with model_path.open("wb") as fh:
        pickle.dump(bundle, fh)

    return {
        "name": ORGAN_CONFIG[organ]["name"],
        "model_path": str(model_path),
        "model_name": bundle["model_name"],
        "train_count": len(train_records),
        "test": _evaluate(pipeline, splits["test"]),
        "val": _evaluate(pipeline, splits["val"]),
        "train_label_distribution": dict(sorted(Counter(y_train).items())),
    }


def train_all() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_ready": True,
        "model_name": "TF-IDF + LogisticRegression",
        "organs": {},
    }
    for organ in ORGAN_CONFIG:
        print(f"Training {organ}...")
        metrics["organs"][organ] = train_one_organ(organ)

    metrics_path = MODELS_DIR / "classifier_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    metrics = train_all()
    print("分类模型训练完成。")
    for organ, item in metrics["organs"].items():
        test_metrics = item["test"]
        print(
            f"- {item['name']}({organ}): "
            f"test accuracy={test_metrics['accuracy']}, "
            f"macro_f1={test_metrics['macro_f1']}, "
            f"model={item['model_path']}"
        )


if __name__ == "__main__":
    main()
