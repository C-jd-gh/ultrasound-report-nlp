from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "thyroid": ROOT_DIR / "USData" / "new_Thyroid2.json",
    "mammary": ROOT_DIR / "USData" / "new_Mammary2.json",
    "liver": ROOT_DIR / "USData" / "new_Liver2.json",
}

FEATURE_PATTERNS = {
    "未见占位": r"未见(?:明确|明显)?占位",
    "结节": r"结节",
    "多发病灶": r"多发|多个|均可见|各见",
    "囊性/无回声": r"囊性|无回声",
    "低回声": r"低回声",
    "术后/切除": r"术后|切除",
    "淋巴结": r"淋巴结|淋巴门",
    "回声不均": r"欠均匀|不均匀|紊乱",
    "回声均匀": r"回声均匀",
    "CDFI": r"CDFI",
    "测量占位符": r"_(?:2DS|3DS|SCM|Loc|LocR|r)_",
    "脂肪肝样描述": r"细密增强|肝肾回声对比增强|肝肾回声反差大",
    "后场衰减": r"后场(?:声|回声)(?:稍)?衰减|后方回声衰减",
    "胆囊结石样": r"声影.*体位|体位.*移动",
    "胆囊息肉样": r"不随体位|彗星尾",
    "餐后胆囊": r"餐后胆囊",
    "分器官冒号模板": r"肝脏：|胆囊：|胰腺：|脾脏：",
}


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [row for rows in payload.values() for row in rows]


def top_ngrams(texts: list[str], labels: list[int], label: int, limit: int = 8) -> list[str]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    binary_target = [1 if value == label else 0 for value in labels]
    scores, _ = chi2(matrix, binary_target)
    terms = vectorizer.get_feature_names_out()
    return [terms[index] for index in scores.argsort()[::-1][:limit]]


def analyze_file(path: Path) -> dict:
    rows = load_rows(path)
    texts = [row["finding"] for row in rows]
    labels = [int(row["labels"]) for row in rows]

    by_text: dict[str, list[int]] = defaultdict(list)
    for text, label in zip(texts, labels):
        by_text[re.sub(r"\s+", "", text)].append(label)

    cross_label_duplicates = {
        text: values for text, values in by_text.items() if len(set(values)) > 1
    }

    result = {
        "total": len(rows),
        "unique_texts": len(by_text),
        "cross_label_duplicate_count": len(cross_label_duplicates),
        "labels": {},
    }
    for label in sorted(set(labels)):
        subset = [text for text, value in zip(texts, labels) if value == label]
        feature_rates = {
            name: round(
                sum(bool(re.search(pattern, text)) for text in subset) / len(subset),
                4,
            )
            for name, pattern in FEATURE_PATTERNS.items()
        }
        representative, frequency = Counter(subset).most_common(1)[0]
        result["labels"][str(label)] = {
            "count": len(subset),
            "top_character_ngrams": top_ngrams(texts, labels, label),
            "feature_rates": feature_rates,
            "representative_text": representative,
            "representative_frequency": frequency,
        }
    return result


def main() -> None:
    report = {organ: analyze_file(path) for organ, path in DATA_FILES.items()}
    output_path = ROOT_DIR / "processed_data" / "label_analysis.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"标签分析完成：{output_path}")
    for organ, item in report.items():
        print(
            f"- {organ}: {item['total']} 条，"
            f"{len(item['labels'])} 个标签，"
            f"跨标签重复文本 {item['cross_label_duplicate_count']} 组"
        )


if __name__ == "__main__":
    main()
