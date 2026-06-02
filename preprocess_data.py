from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from nlp_pipeline import (
    DATA_DIR,
    ORGAN_CONFIG,
    PLACEHOLDER_MAP,
    PROCESSED_DATA_DIR,
    build_dictionary,
    clean_text,
    dataset_stats,
    load_corpus,
    tokenize,
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def preprocess_item(item: dict) -> tuple[dict, dict]:
    cleaned = deepcopy(item)
    original = str(item.get("finding", ""))
    finding = clean_text(original)
    placeholders = re.findall(r"_[A-Za-z0-9]+_", finding)
    cleaned["finding"] = finding
    if original != finding:
        cleaned["original_finding"] = original
    cleaned["placeholder_tags"] = placeholders
    cleaned["token_preview"] = tokenize(finding)[:40]
    return cleaned, {
        "changed": original != finding,
        "placeholders": Counter(placeholders),
        "length": len(finding),
    }


def preprocess_dataset() -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dir": str(DATA_DIR),
        "output_dir": str(PROCESSED_DATA_DIR),
        "placeholder_map": PLACEHOLDER_MAP,
        "organs": {},
    }

    for organ, cfg in ORGAN_CONFIG.items():
        raw_path = DATA_DIR / cfg["file"]
        output_path = PROCESSED_DATA_DIR / cfg["processed_file"]
        raw_payload = read_json(raw_path)
        cleaned_payload = {}
        organ_placeholder_counter: Counter[str] = Counter()
        split_counts = {}
        changed_count = 0
        total_length = 0

        for split, items in raw_payload.items():
            cleaned_items = []
            for item in items:
                cleaned_item, meta = preprocess_item(item)
                cleaned_items.append(cleaned_item)
                changed_count += int(meta["changed"])
                organ_placeholder_counter.update(meta["placeholders"])
                total_length += meta["length"]
            cleaned_payload[split] = cleaned_items
            split_counts[split] = len(cleaned_items)

        total = sum(split_counts.values())
        write_json(output_path, cleaned_payload)
        summary["organs"][organ] = {
            "name": cfg["name"],
            "raw_file": str(raw_path),
            "clean_file": str(output_path),
            "total": total,
            "splits": split_counts,
            "changed_reports": changed_count,
            "average_length": round(total_length / total, 2) if total else 0,
            "placeholders": dict(organ_placeholder_counter.most_common()),
        }

    # The files now exist, so clear cached raw reads before generating derived outputs.
    load_corpus.cache_clear()
    build_dictionary.cache_clear()
    dataset_stats.cache_clear() if hasattr(dataset_stats, "cache_clear") else None

    write_json(PROCESSED_DATA_DIR / "preprocess_summary.json", summary)
    write_json(PROCESSED_DATA_DIR / "dataset_stats.json", dataset_stats())
    write_json(
        PROCESSED_DATA_DIR / "medical_dictionary.json",
        {
            "size": len(build_dictionary()),
            "terms": list(build_dictionary()),
        },
    )
    return summary


def main() -> None:
    summary = preprocess_dataset()
    print("数据预处理完成。")
    for organ, item in summary["organs"].items():
        print(
            f"- {item['name']}({organ}): {item['total']} 条，"
            f"清洗变化 {item['changed_reports']} 条，输出 {item['clean_file']}"
        )
    print(f"输出目录：{PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
