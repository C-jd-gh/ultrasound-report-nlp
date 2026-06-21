from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultrasound_nlp.nlp_pipeline import (
    PROCESSED_DATA_DIR,
    SEGMENTATION_METRICS_PATH,
    evaluate_segmentation,
)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_segmentation()
    SEGMENTATION_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"分词评测完成：{SEGMENTATION_METRICS_PATH}")
    for name, item in metrics["overall"].items():
        print(
            f"- {name}: precision={item['precision']}, "
            f"recall={item['recall']}, f1={item['f1']}"
        )


if __name__ == "__main__":
    main()
