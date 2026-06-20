from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultrasound_nlp.sequence_ner import train_all


def main() -> None:
    metrics = train_all()
    print("CRF 实体识别模型训练完成。")
    for organ, item in metrics["organs"].items():
        print(
            f"- {item['name']}({organ}): "
            f"train={item['train_count']}, test={item['test_count']}, "
            f"weighted_f1={item['weighted_f1']}, model={item['model_path']}"
        )


if __name__ == "__main__":
    main()
