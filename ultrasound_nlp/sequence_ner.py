from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

from .nlp_pipeline import (
    DOMAIN_TERMS,
    MODELS_DIR,
    ORGAN_CONFIG,
    REGEX_RULES,
    clean_text,
    load_base_dictionary,
    records_for_organ,
)


ENTITY_LABELS = {
    "organ": "ORGAN",
    "location": "LOCATION",
    "lesion": "LESION",
    "echo": "ECHO",
    "boundary": "BOUNDARY",
    "shape": "SHAPE",
    "blood": "BLOOD",
    "status": "STATUS",
    "measurement": "MEASUREMENT",
    "lymph": "LYMPH",
    "surgery": "SURGERY",
}


def model_path(organ: str) -> Path:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    return MODELS_DIR / f"{organ}_sequence_ner.pkl"


def metrics_path() -> Path:
    return MODELS_DIR / "ner_metrics.json"


def _regex_label(rule_type: str) -> str:
    if rule_type == "lymph":
        return "LYMPH"
    if rule_type == "surgery":
        return "SURGERY"
    return ENTITY_LABELS.get(rule_type, rule_type.upper())


def weak_spans(text: str) -> list[dict[str, Any]]:
    text = clean_text(text)
    spans: list[dict[str, Any]] = []
    dictionary = load_base_dictionary()

    for entity_type, terms in dictionary.items():
        label = ENTITY_LABELS.get(entity_type)
        if not label:
            continue
        for term in sorted(terms, key=len, reverse=True):
            if not term:
                continue
            for match in re.finditer(re.escape(term), text):
                spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "text": match.group(0),
                        "source": "weak_dictionary",
                    }
                )

    for rule_type, patterns in REGEX_RULES.items():
        label = _regex_label(rule_type)
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "text": match.group(0),
                        "source": "weak_regex",
                    }
                )

    spans.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    selected: list[dict[str, Any]] = []
    occupied = [False] * len(text)
    for span in spans:
        if span["start"] >= span["end"]:
            continue
        if any(occupied[pos] for pos in range(span["start"], span["end"])):
            continue
        for pos in range(span["start"], span["end"]):
            occupied[pos] = True
        selected.append(span)
    selected.sort(key=lambda item: (item["start"], item["end"]))
    return selected


def weak_bio_labels(text: str) -> list[str]:
    text = clean_text(text)
    labels = ["O"] * len(text)
    for span in weak_spans(text):
        labels[span["start"]] = f"B-{span['label']}"
        for pos in range(span["start"] + 1, span["end"]):
            labels[pos] = f"I-{span['label']}"
    return labels


def _char_type(char: str) -> str:
    if char.isdigit():
        return "digit"
    if char.isascii() and char.isalpha():
        return "alpha"
    if char == "_":
        return "placeholder_mark"
    if "\u4e00" <= char <= "\u9fff":
        return "han"
    return "punct"


def char_features(text: str, index: int) -> dict[str, Any]:
    char = text[index]
    prev_char = text[index - 1] if index > 0 else "<BOS>"
    next_char = text[index + 1] if index < len(text) - 1 else "<EOS>"
    window_left = text[max(0, index - 2) : index]
    window_right = text[index + 1 : index + 3]
    return {
        "bias": 1.0,
        "char": char,
        "type": _char_type(char),
        "prev": prev_char,
        "next": next_char,
        "left2": window_left,
        "right2": window_right,
        "is_digit": char.isdigit(),
        "is_ascii": char.isascii(),
        "is_placeholder_area": "_" in text[max(0, index - 5) : min(len(text), index + 6)],
        "has_cdfi_near": "CDFI" in text[max(0, index - 8) : min(len(text), index + 8)],
        "has_blood_near": "血流" in text[max(0, index - 8) : min(len(text), index + 8)],
        "has_echo_near": "回声" in text[max(0, index - 8) : min(len(text), index + 8)],
        "has_lesion_near": any(term in text[max(0, index - 8) : min(len(text), index + 8)] for term in DOMAIN_TERMS["lesion"]),
    }


def sentence_features(text: str) -> list[dict[str, Any]]:
    text = clean_text(text)
    return [char_features(text, index) for index in range(len(text))]


def _records_to_xy(records) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    x_rows = []
    y_rows = []
    for record in records:
        text = clean_text(record.finding)
        if not text:
            continue
        x_rows.append(sentence_features(text))
        y_rows.append(weak_bio_labels(text))
    return x_rows, y_rows


def _bio_to_entities(text: str, labels: list[str]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    start = None
    current_label = None
    for index, label in enumerate(labels + ["O"]):
        if label.startswith("B-"):
            if current_label is not None and start is not None:
                entities.append(
                    {
                        "type": current_label.lower(),
                        "label": current_label,
                        "value": text[start:index],
                        "start": start,
                        "end": index,
                        "source": "crf",
                    }
                )
            start = index
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            continue
        else:
            if current_label is not None and start is not None:
                entities.append(
                    {
                        "type": current_label.lower(),
                        "label": current_label,
                        "value": text[start:index],
                        "start": start,
                        "end": index,
                        "source": "crf",
                    }
                )
            start = None
            current_label = None
    return [item for item in entities if item["value"]]


def train_one_organ(organ: str) -> dict[str, Any]:
    records = records_for_organ(organ)
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    if not train_records:
        raise ValueError(f"No training records found for {organ}")

    x_train, y_train = _records_to_xy(train_records)
    x_test, y_test = _records_to_xy(test_records)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=80,
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)

    y_pred = crf.predict(x_test) if x_test else []
    labels = sorted(label for label in crf.classes_ if label != "O")
    metrics = {
        "name": ORGAN_CONFIG[organ]["name"],
        "model_path": str(model_path(organ)),
        "train_count": len(train_records),
        "test_count": len(test_records),
        "label_distribution": dict(Counter(label for row in y_train for label in row)),
        "weighted_f1": round(float(flat_f1_score(y_test, y_pred, average="weighted", labels=labels)), 4) if y_test else None,
        "classification_report": flat_classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0) if y_test else {},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with model_path(organ).open("wb") as fh:
        pickle.dump({"organ": organ, "model_name": "Weak BIO + CRF", "crf": crf}, fh)
    return metrics


def train_all() -> dict[str, Any]:
    payload = {"model_ready": True, "model_name": "Weak BIO + CRF", "organs": {}}
    for organ in ORGAN_CONFIG:
        payload["organs"][organ] = train_one_organ(organ)
    with metrics_path().open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def load_metrics() -> dict[str, Any]:
    path = metrics_path()
    if not path.exists():
        return {
            "model_ready": False,
            "message": "CRF 实体识别模型尚未训练，请先运行 python scripts/train_ner.py",
            "organs": {},
        }
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["model_ready"] = True
    return payload


def predict_sequence_entities(text: str, organ: str = "thyroid") -> dict[str, Any]:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    path = model_path(organ)
    if not path.exists():
        return {
            "model_ready": False,
            "message": "CRF 实体识别模型尚未训练，请先运行 python scripts/train_ner.py",
            "entities": [],
        }
    cleaned = clean_text(text)
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    labels = bundle["crf"].predict_single(sentence_features(cleaned))
    return {
        "model_ready": True,
        "model_name": bundle.get("model_name", "Weak BIO + CRF"),
        "entities": _bio_to_entities(cleaned, labels),
    }
