from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultrasound_nlp.vector_models import train_all_vectors


def main() -> None:
    metrics = train_all_vectors()
    print("向量与聚类模型训练完成。")
    print(
        f"- Word2Vec: vocab={metrics['word2vec']['vocabulary_size']}, "
        f"model={metrics['word2vec']['model_path']}"
    )
    for organ, item in metrics["clusters"]["organs"].items():
        print(
            f"- {item['name']}({organ}): "
            f"samples={item['sample_count']}, clusters={item['cluster_count']}, "
            f"silhouette={item['silhouette_score']}, model={item['model_path']}"
        )


if __name__ == "__main__":
    main()
