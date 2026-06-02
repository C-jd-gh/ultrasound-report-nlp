from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - the app has a non-sklearn fallback
    TfidfVectorizer = None
    cosine_similarity = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "USData"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"


ORGAN_CONFIG = {
    "thyroid": {
        "name": "甲状腺",
        "file": "new_Thyroid2.json",
        "processed_file": "thyroid_clean.json",
        "organs": ["甲状腺", "腺体", "峡部", "颈部"],
    },
    "mammary": {
        "name": "乳腺",
        "file": "new_Mammary2.json",
        "processed_file": "mammary_clean.json",
        "organs": ["乳腺", "乳房", "乳导管", "腋下", "乳头"],
    },
    "liver": {
        "name": "肝胆胰脾",
        "file": "new_Liver2.json",
        "processed_file": "liver_clean.json",
        "organs": ["肝脏", "胆囊", "胰腺", "脾脏", "门静脉", "胆管"],
    },
}


PLACEHOLDER_MAP = {
    "_SMM_": "毫米数值",
    "_SCM_": "厘米数值",
    "_Loc_": "钟点/方位",
    "_LocR_": "范围方位",
    "_2DS_": "二维尺寸",
    "_3DS_": "三维尺寸",
    "_r_": "纵横比",
}


DOMAIN_TERMS = {
    "organ": [
        "甲状腺",
        "乳腺",
        "乳房",
        "乳导管",
        "肝脏",
        "胆囊",
        "胰腺",
        "脾脏",
        "门静脉",
        "胆管",
        "淋巴结",
        "腺体",
    ],
    "location": [
        "左叶",
        "右叶",
        "峡部",
        "上极",
        "下极",
        "中部",
        "中上段",
        "中下段",
        "右前叶",
        "右后叶",
        "左内叶",
        "肝右叶",
        "肝左叶",
        "双侧",
        "左侧",
        "右侧",
        "腋下",
        "颈部",
        "乳头",
        "皮下",
        "脂肪层",
        "外上象限",
        "内上象限",
        "外下象限",
        "内下象限",
    ],
    "lesion": [
        "结节",
        "低回声结节",
        "无回声结节",
        "强回声结节",
        "囊实混合回声结节",
        "囊性结构",
        "囊肿",
        "占位性病变",
        "钙化",
        "强回声",
        "低回声区",
        "片状低回声区",
        "肿大淋巴结",
    ],
    "echo": [
        "低回声",
        "高回声",
        "强回声",
        "无回声",
        "等回声",
        "混合回声",
        "囊实混合回声",
        "实性回声",
        "回声均匀",
        "回声欠均匀",
        "回声不均匀",
        "实质回声细密增强",
        "后方回声增强",
        "声影",
        "彗星尾",
    ],
    "boundary": [
        "边界清晰",
        "边界清楚",
        "边界尚清晰",
        "边界欠清晰",
        "边界不清晰",
        "包膜光滑",
        "壁不厚",
        "壁稍厚",
        "欠光滑",
    ],
    "shape": [
        "形态规整",
        "形态规则",
        "形态尚规整",
        "形态欠规整",
        "形态不规则",
        "类圆形",
        "椭圆形",
        "分叶状",
        "毛刺",
        "纵横比",
    ],
    "blood": [
        "CDFI",
        "血流信号",
        "未探及血流信号",
        "未见异常血流信号",
        "可探及血流信号",
        "血流丰富",
        "血流信号稍增多",
        "血流信号分布正常",
    ],
    "status": [
        "未见明确占位性病变",
        "未见明显异常回声",
        "未见明显肿大淋巴结",
        "术后",
        "切除术后",
        "全切术后",
        "显示欠清晰",
        "肠气干扰",
    ],
    "measurement": list(PLACEHOLDER_MAP),
}


STOPWORDS = {
    "的",
    "于",
    "可",
    "见",
    "未",
    "示",
    "内",
    "外",
    "和",
    "及",
    "其",
    "余",
    "为",
    "约",
    "如常",
    "明显",
    "明确",
    "大小",
    "形态",
    "检查",
    "扫查",
}


SYNONYM_RULES = [
    ("边界清楚", "边界清晰"),
    ("边界尚清", "边界尚清晰"),
    ("边界欠清楚", "边界欠清晰"),
    ("形态规则", "形态规整"),
    ("形态尚规则", "形态尚规整"),
    ("未探及", "未见"),
    ("未记录到", "未见"),
    ("无明显", "未见明显"),
    ("未见明确", "未见明显"),
    ("不扩张", "未见扩张"),
    ("不宽", "未见增宽"),
]


REGEX_RULES = {
    "measurement": [
        r"_(?:2DS|3DS|SCM|SMM|Loc|LocR|r)_",
        r"\d+(?:\.\d+)?\s*(?:cm|mm|厘米|毫米)",
    ],
    "location": [
        r"(?:左|右|双)侧(?:乳腺|腋下|颈部)?",
        r"(?:左|右)叶(?:上极|下极|中部|中上段|中下段|下段|上段)?",
        r"(?:肝右叶|肝左叶|右前叶|右后叶|左内叶|峡部|腋下|乳头|胆囊旁)",
    ],
    "blood": [r"CDFI[^。；;，,]*?(?:血流信号|血流|未见|未探及)"],
    "boundary": [r"边界(?:清晰|清楚|尚清晰|欠清晰|不清晰|不清楚)", r"包膜(?:光滑|欠光滑)"],
    "shape": [r"形态(?:规整|规则|尚规整|欠规整|不规则)", r"纵横比[^，。；;]*", r"(?:类圆形|椭圆形|分叶状|毛刺)"],
    "echo": [r"(?:低|高|强|等|无|混合|囊实混合|实性|偏低|偏强)回声(?:结节|区)?", r"回声(?:均匀|欠均匀|不均匀|细密增强|增粗|减低)"],
    "lymph": [r"(?:淋巴结|淋巴门|肿大淋巴结)[^。；;]*"],
    "surgery": [r"[^。；;]*(?:术后|切除|全切)[^。；;]*"],
    "lesion": [r"(?:结节|囊肿|囊性结构|占位性病变|钙化|低回声区|强回声团)[^。；;，,]*"],
}


@dataclass(frozen=True)
class CorpusRecord:
    uid: int
    organ: str
    organ_name: str
    split: str
    label: int
    finding: str


def _repair_mojibake(text: str) -> str:
    suspicious = sum(text.count(ch) for ch in "åäæçèéï¼ã€")
    if suspicious < 3:
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if len(fixed) >= len(text) * 0.8 else text


def clean_text(text: str) -> str:
    text = _repair_mojibake(str(text or ""))
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("；", "；").replace(";", "；")
    text = text.replace(":", "：")
    return text.strip()


@lru_cache(maxsize=1)
def load_corpus() -> tuple[CorpusRecord, ...]:
    records: list[CorpusRecord] = []
    for organ, cfg in ORGAN_CONFIG.items():
        processed_path = PROCESSED_DATA_DIR / cfg["processed_file"]
        path = processed_path if processed_path.exists() else DATA_DIR / cfg["file"]
        with path.open("r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
        for split, items in payload.items():
            for item in items:
                records.append(
                    CorpusRecord(
                        uid=int(item.get("uid", 0)),
                        organ=organ,
                        organ_name=cfg["name"],
                        split=str(item.get("split") or split),
                        label=int(item.get("labels", -1)),
                        finding=clean_text(item.get("finding", "")),
                    )
                )
    return tuple(records)


def records_for_organ(organ: str | None = None) -> list[CorpusRecord]:
    records = list(load_corpus())
    if organ and organ in ORGAN_CONFIG:
        records = [record for record in records if record.organ == organ]
    return records


@lru_cache(maxsize=1)
def load_base_dictionary() -> dict[str, list[str]]:
    categories = {key: set(values) for key, values in DOMAIN_TERMS.items()}
    word_file = DATA_DIR / "key_technical_words.txt"
    if word_file.exists():
        for line in word_file.read_text(encoding="utf-8-sig").splitlines():
            word = line.strip()
            if not word:
                continue
            categories.setdefault("technical", set()).add(word)
    for cfg in ORGAN_CONFIG.values():
        categories.setdefault("organ", set()).update(cfg["organs"])
    for placeholder, name in PLACEHOLDER_MAP.items():
        categories.setdefault("measurement", set()).add(placeholder)
        categories.setdefault("measurement", set()).add(name)
    return {key: sorted(values, key=lambda item: (-len(item), item)) for key, values in categories.items()}


@lru_cache(maxsize=1)
def build_dictionary() -> tuple[str, ...]:
    words: set[str] = set()
    for values in load_base_dictionary().values():
        words.update(values)

    # Add frequent report-specific phrases so the dictionary grows from data.
    counts: Counter[str] = Counter()
    for record in load_corpus():
        for segment in re.findall(r"[\u4e00-\u9fffA-Za-z_]{2,}", record.finding):
            for size in range(2, min(7, len(segment) + 1)):
                for idx in range(0, len(segment) - size + 1):
                    gram = segment[idx : idx + size]
                    if not any(stop in gram for stop in ["未见", "可见", "大小约"]):
                        counts[gram] += 1
    for word, count in counts.items():
        if count >= 12 and any(key in word for key in ["回声", "结节", "血流", "边界", "形态", "淋巴", "胆囊", "腺体"]):
            words.add(word)
    return tuple(sorted(words, key=lambda item: (-len(item), item)))


def _scan_alnum(text: str, start: int) -> tuple[str, int]:
    match = re.match(r"_[A-Za-z0-9]+_|[A-Za-z0-9]+", text[start:])
    if match:
        return match.group(0), start + len(match.group(0))
    return text[start], start + 1


def forward_max_match(text: str, max_len: int = 8) -> list[str]:
    dictionary = set(build_dictionary())
    tokens: list[str] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if re.match(r"[\s，。；、：:,.!?！？（）()“”\"']", char):
            idx += 1
            continue
        if char == "_" or char.isascii() and char.isalnum():
            token, idx = _scan_alnum(text, idx)
            tokens.append(token)
            continue
        end = min(len(text), idx + max_len)
        found = ""
        for pos in range(end, idx, -1):
            candidate = text[idx:pos]
            if candidate in dictionary:
                found = candidate
                break
        if found:
            if found[:1] in {"一", "多", "数"} and found[1:] in dictionary:
                found = found[1:]
            tokens.append(found)
            idx += len(found)
        else:
            tokens.append(char)
            idx += 1
    return [token for token in tokens if token and token not in STOPWORDS]


def reverse_max_match(text: str, max_len: int = 8) -> list[str]:
    dictionary = set(build_dictionary())
    tokens: list[str] = []
    idx = len(text)
    while idx > 0:
        char = text[idx - 1]
        if re.match(r"[\s，。；、：:,.!?！？（）()“”\"']", char):
            idx -= 1
            continue
        if char == "_" or char.isascii() and char.isalnum():
            start = idx - 1
            while start > 0 and re.match(r"[A-Za-z0-9_]", text[start - 1]):
                start -= 1
            tokens.append(text[start:idx])
            idx = start
            continue
        start = max(0, idx - max_len)
        found = ""
        for pos in range(start, idx):
            candidate = text[pos:idx]
            if candidate in dictionary:
                found = candidate
                break
        if found:
            if found[:1] in {"一", "多", "数"} and found[1:] in dictionary:
                found = found[1:]
            tokens.append(found)
            idx -= len(found)
        else:
            tokens.append(char)
            idx -= 1
    return [token for token in reversed(tokens) if token and token not in STOPWORDS]


def tokenize(text: str, mode: str = "forward") -> list[str]:
    text = clean_text(text)
    if mode == "reverse":
        return reverse_max_match(text)
    forward = forward_max_match(text)
    reverse = reverse_max_match(text)
    if mode == "both":
        return sorted(set(forward) | set(reverse), key=lambda token: (-len(token), token))
    # Choose the result with fewer single-character fragments.
    f_single = sum(1 for token in forward if len(token) == 1)
    r_single = sum(1 for token in reverse if len(token) == 1)
    return reverse if r_single < f_single else forward


def normalize_text(text: str) -> str:
    normalized = clean_text(text)
    for source, target in SYNONYM_RULES:
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("_2DS_", "二维尺寸")
    normalized = normalized.replace("_3DS_", "三维尺寸")
    normalized = normalized.replace("_SCM_", "厘米数值")
    normalized = normalized.replace("_SMM_", "毫米数值")
    normalized = normalized.replace("_Loc_", "方位")
    normalized = normalized.replace("_LocR_", "范围方位")
    normalized = normalized.replace("_r_", "纵横比")
    return normalized


def extract_regex_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    text = clean_text(text)
    for rule_type, patterns in REGEX_RULES.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                matches.append(
                    {
                        "type": rule_type,
                        "value": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "rule": pattern,
                    }
                )
    matches.sort(key=lambda item: (item["start"], item["end"], item["type"]))
    return _dedupe_dicts(matches, ("type", "value", "start", "end"))


def _dedupe_dicts(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        marker = tuple(item.get(key) for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def _sentence_parts(text: str) -> list[str]:
    return [part for part in re.split(r"[。；;]", clean_text(text)) if part]


def extract_entities(text: str, organ: str = "thyroid") -> list[dict[str, Any]]:
    text = clean_text(text)
    categories = load_base_dictionary()
    entities: list[dict[str, Any]] = []

    for entity_type, terms in categories.items():
        if entity_type == "technical":
            continue
        for term in terms:
            if term and term in text:
                entities.append(
                    {
                        "type": entity_type,
                        "name": term,
                        "value": term,
                        "source": "dictionary",
                        "evidence": term,
                    }
                )

    for match in extract_regex_matches(text):
        entities.append(
            {
                "type": match["type"],
                "name": match["type"],
                "value": match["value"],
                "source": "regex",
                "evidence": match["value"],
            }
        )

    lesion_terms = DOMAIN_TERMS["lesion"]
    for sentence in _sentence_parts(text):
        if not any(term in sentence for term in lesion_terms):
            continue
        entities.append(
            {
                "type": "lesion_event",
                "name": _first_match(sentence, lesion_terms) or "病灶",
                "value": sentence,
                "source": "sentence_rule",
                "evidence": sentence,
                "attributes": {
                    "location": _first_regex(sentence, REGEX_RULES["location"]),
                    "size": _first_regex(sentence, REGEX_RULES["measurement"]),
                    "echo": _first_regex(sentence, REGEX_RULES["echo"]),
                    "boundary": _first_regex(sentence, REGEX_RULES["boundary"]),
                    "shape": _first_regex(sentence, REGEX_RULES["shape"]),
                    "blood": _first_regex(sentence, REGEX_RULES["blood"]),
                },
            }
        )

    return _dedupe_dicts(entities, ("type", "value", "source"))


def _first_match(text: str, terms: list[str]) -> str | None:
    sorted_terms = sorted(terms, key=len, reverse=True)
    return next((term for term in sorted_terms if term in text), None)


def _first_regex(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def keyword_by_frequency(tokens: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    counter = Counter(token for token in tokens if len(token) > 1 and token not in STOPWORDS)
    return [{"word": word, "score": count} for word, count in counter.most_common(top_k)]


@lru_cache(maxsize=4)
def _tfidf_model(organ: str):
    if TfidfVectorizer is None:
        return None, None, []
    records = records_for_organ(organ)
    docs = [record.finding for record in records]
    vectorizer = TfidfVectorizer(
        tokenizer=lambda value: tokenize(value),
        token_pattern=None,
        lowercase=False,
        min_df=1,
        max_features=3000,
    )
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix, records


def keyword_by_tfidf(text: str, organ: str, top_k: int = 12) -> list[dict[str, Any]]:
    model = _tfidf_model(organ)
    vectorizer, _, _ = model
    if vectorizer is None:
        tokens = tokenize(text)
        total = len(tokens) or 1
        return [{"word": word, "score": round(count / total, 4)} for word, count in Counter(tokens).most_common(top_k)]
    vector = vectorizer.transform([clean_text(text)])
    names = vectorizer.get_feature_names_out()
    row = vector.toarray()[0]
    ranked = sorted(((names[idx], row[idx]) for idx in row.nonzero()[0]), key=lambda item: item[1], reverse=True)
    return [{"word": word, "score": round(float(score), 4)} for word, score in ranked[:top_k]]


def similar_reports(text: str, organ: str, limit: int = 5) -> list[dict[str, Any]]:
    vectorizer, matrix, records = _tfidf_model(organ)
    if vectorizer is None or matrix is None or cosine_similarity is None:
        return _similar_reports_fallback(text, organ, limit)
    query = vectorizer.transform([clean_text(text)])
    scores = cosine_similarity(query, matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
    return [
        {
            "uid": records[idx].uid,
            "split": records[idx].split,
            "label": records[idx].label,
            "score": round(float(score), 4),
            "finding": records[idx].finding,
        }
        for idx, score in ranked
    ]


def _similar_reports_fallback(text: str, organ: str, limit: int) -> list[dict[str, Any]]:
    query = set(tokenize(text))
    rows = []
    for record in records_for_organ(organ):
        tokens = set(tokenize(record.finding))
        score = len(query & tokens) / math.sqrt((len(query) or 1) * (len(tokens) or 1))
        rows.append((score, record))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "uid": record.uid,
            "split": record.split,
            "label": record.label,
            "score": round(score, 4),
            "finding": record.finding,
        }
        for score, record in rows[:limit]
    ]


def quality_check(text: str, entities: list[dict[str, Any]]) -> list[dict[str, str]]:
    text = clean_text(text)
    warnings: list[dict[str, str]] = []
    has_lesion = any(term in text for term in ["结节", "囊肿", "占位", "钙化", "低回声区", "囊性结构"])
    has_size = bool(re.search(r"_(?:2DS|3DS|SCM|SMM)_|\d+(?:\.\d+)?\s*(?:cm|mm|厘米|毫米)", text))
    has_location = any(entity["type"] == "location" for entity in entities)
    has_blood = "CDFI" in text or "血流" in text
    has_boundary = "边界" in text
    has_shape = "形态" in text

    if has_lesion and not has_size:
        warnings.append({"level": "warning", "message": "发现病灶描述，但缺少尺寸信息。"})
    if has_lesion and not has_location:
        warnings.append({"level": "warning", "message": "发现病灶描述，但缺少明确部位。"})
    if has_lesion and not has_blood:
        warnings.append({"level": "info", "message": "病灶描述中未出现 CDFI 或血流信息。"})
    if has_lesion and not has_boundary:
        warnings.append({"level": "info", "message": "病灶描述中未出现边界信息。"})
    if has_lesion and not has_shape:
        warnings.append({"level": "info", "message": "病灶描述中未出现形态信息。"})
    if not warnings:
        warnings.append({"level": "ok", "message": "报告主要结构完整，未发现明显缺项。"})
    return warnings


def generate_template_report(text: str, organ: str, entities: list[dict[str, Any]]) -> str:
    organ_name = ORGAN_CONFIG.get(organ, ORGAN_CONFIG["thyroid"])["name"]
    lesion_events = [entity for entity in entities if entity["type"] == "lesion_event"]
    normalized = normalize_text(text)

    normal_phrases = []
    if "未见明显占位性病变" in normalized or "未见占位" in normalized:
        normal_phrases.append("未见明显占位性病变")
    if "未见明显肿大淋巴结" in normalized or "未见异常肿大淋巴结" in normalized:
        normal_phrases.append("未见明显肿大淋巴结")
    if "未见异常血流信号" in normalized or "未见血流信号" in normalized:
        normal_phrases.append("未见异常血流信号")

    lines = [f"{organ_name}超声结构化摘要："]
    if lesion_events:
        lines.append(f"共识别到 {len(lesion_events)} 条病灶相关描述。")
        for idx, event in enumerate(lesion_events[:4], 1):
            attrs = event.get("attributes", {})
            parts = [
                attrs.get("location") or "部位未明确",
                attrs.get("echo") or event.get("name") or "病灶",
                attrs.get("size") or "尺寸未明确",
                attrs.get("boundary") or "边界未明确",
                attrs.get("shape") or "形态未明确",
                attrs.get("blood") or "血流未明确",
            ]
            lines.append(f"{idx}. " + "，".join(parts) + "。")
    else:
        lines.append("未抽取到明确病灶句，按正常或弥漫性描述处理。")

    if normal_phrases:
        lines.append("其他描述：" + "；".join(dict.fromkeys(normal_phrases)) + "。")
    lines.append("以上为模板化文本摘要，仅用于 NLP 课程演示。")
    return "\n".join(lines)


def analyze_report(text: str, organ: str = "thyroid") -> dict[str, Any]:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    cleaned = clean_text(text)
    tokens_forward = forward_max_match(cleaned)
    tokens_reverse = reverse_max_match(cleaned)
    tokens = tokenize(cleaned)
    regex_matches = extract_regex_matches(cleaned)
    entities = extract_entities(cleaned, organ)
    normalized = normalize_text(cleaned)
    return {
        "organ": organ,
        "organ_name": ORGAN_CONFIG[organ]["name"],
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "tokens_forward": tokens_forward,
        "tokens_reverse": tokens_reverse,
        "keywords_frequency": keyword_by_frequency(tokens),
        "keywords_tfidf": keyword_by_tfidf(cleaned, organ),
        "regex_matches": regex_matches,
        "entities": entities,
        "normalized": normalized,
        "template_report": generate_template_report(cleaned, organ, entities),
        "quality": quality_check(cleaned, entities),
    }


def dataset_stats() -> dict[str, Any]:
    records = load_corpus()
    by_organ: dict[str, Any] = {}
    for organ, cfg in ORGAN_CONFIG.items():
        subset = [record for record in records if record.organ == organ]
        split_counter = Counter(record.split for record in subset)
        label_counter = Counter(record.label for record in subset)
        placeholder_counter: Counter[str] = Counter()
        token_counter: Counter[str] = Counter()
        for record in subset:
            placeholder_counter.update(re.findall(r"_[A-Za-z0-9]+_", record.finding))
            token_counter.update(token for token in tokenize(record.finding) if len(token) > 1)
        by_organ[organ] = {
            "name": cfg["name"],
            "total": len(subset),
            "splits": dict(split_counter),
            "labels": dict(sorted(label_counter.items())),
            "placeholders": dict(placeholder_counter.most_common()),
            "top_terms": [{"word": word, "count": count} for word, count in token_counter.most_common(20)],
        }
    return {
        "total": len(records),
        "organs": by_organ,
        "dictionary_size": len(build_dictionary()),
        "placeholder_map": PLACEHOLDER_MAP,
    }


def sample_report(organ: str = "thyroid", index: int = 0) -> dict[str, Any]:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    records = records_for_organ(organ)
    record = records[index % len(records)]
    return {
        "uid": record.uid,
        "organ": record.organ,
        "organ_name": record.organ_name,
        "split": record.split,
        "label": record.label,
        "finding": record.finding,
        "index": index % len(records),
        "total": len(records),
    }


def dictionary_view() -> dict[str, Any]:
    categories = load_base_dictionary()
    return {
        "categories": {key: values[:80] for key, values in categories.items()},
        "size": len(build_dictionary()),
        "dynamic_examples": list(build_dictionary())[:80],
    }
