from __future__ import annotations

import json
import math
import re
from io import BytesIO
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jieba
import jieba.posseg as pseg
from wordcloud import WordCloud


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "USData"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
RESOURCES_DIR = BASE_DIR / "resources"
JIEBA_USER_DICT_PATH = PROCESSED_DATA_DIR / "medical_jieba_userdict.txt"
GOLD_DATA_PATH = RESOURCES_DIR / "segmentation_gold.json"
SEGMENTATION_METRICS_PATH = PROCESSED_DATA_DIR / "segmentation_metrics.json"
WORDCLOUD_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)


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
        "甲状腺", "甲状腺体积", "乳腺", "乳房", "乳导管", "肝脏", "胆囊",
        "胰腺", "脾脏", "门静脉", "胆管", "淋巴结", "腺体", "腺体内",
    ],
    "location": [
        "左叶", "右叶", "峡部", "上极", "下极", "中部", "中上段", "中下段",
        "左叶中下部", "右叶中下部", "右叶近峡部", "左叶近峡部", "右前叶",
        "右后叶", "左内叶", "肝右叶", "肝左叶", "双侧", "左侧", "右侧",
        "腋下", "颈部", "双侧颈部", "原区域", "乳头", "皮下", "脂肪层",
        "外上象限", "内上象限", "外下象限", "内下象限",
    ],
    "lesion": [
        "结节", "多发结节", "低回声结节", "无回声结节", "强回声结节",
        "囊实混合回声结节", "囊性为主的囊实混合回声结节",
        "实性为主的囊实混合回声结节", "囊性结构", "囊肿", "占位性病变",
        "钙化", "低回声区", "片状低回声区", "肿大淋巴结",
        "异常肿大淋巴结", "明显肿大淋巴结",
    ],
    "echo": [
        "低回声", "高回声", "强回声", "无回声", "等回声", "混合回声",
        "囊实混合回声", "实性回声", "以囊性为主", "以实性为主",
        "囊性为主", "实性为主", "回声均匀", "回声欠均匀", "回声不均匀",
        "实质回声细密增强", "腺体内回声增粗", "腺体内回声增强",
        "后方回声增强", "后场回声衰减", "回声增粗", "回声增强",
        "不均匀", "声影", "彗星尾",
    ],
    "boundary": [
        "边界清晰", "边界清楚", "边界尚清晰", "边界欠清晰", "边界不清晰",
        "包膜光滑", "壁不厚", "壁稍厚", "欠光滑",
    ],
    "shape": [
        "形态规整", "形态规则", "形态尚规整", "形态欠规整", "形态不规则",
        "类圆形", "椭圆形", "分叶状", "毛刺", "纵横比",
    ],
    "blood": [
        "CDFI", "血流信号", "未探及血流信号", "未见异常血流信号",
        "未见明显异常血流信号", "可探及血流信号", "周边可探及血流信号",
        "内部可探及血流信号", "周边及内部可探及血流信号", "血流丰富",
        "血流信号稍增多", "血流信号分布正常",
    ],
    "status": [
        "大小形态如常", "甲状腺形态饱满", "形态饱满", "甲状腺体积正常",
        "体积正常", "腺体回声均匀", "部分可见条索样改变", "条索样改变",
        "未见明确占位性病变", "未见明显占位性病变",
        "未见明显异常回声", "未见明显异常",
        "未见明显肿大淋巴结", "未见异常肿大淋巴结", "术后", "切除术后",
        "全切术后", "甲状腺全切术后", "显示欠清晰", "肠气干扰",
    ],
    "measurement": list(PLACEHOLDER_MAP),
}


PUNCTUATION_PATTERN = re.compile(r"[\s，。；、：:,.!?！？（）()“”\"'；]+")
PLACEHOLDER_PATTERN = re.compile(r"_[A-Za-z0-9]+_")
ALNUM_PATTERN = re.compile(r"_[A-Za-z0-9]+_|[A-Za-z0-9]+")


@dataclass(frozen=True)
class CorpusRecord:
    uid: int
    organ: str
    organ_name: str
    split: str
    finding: str


def _read_lines(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    return frozenset(_read_lines(RESOURCES_DIR / "stopwords.txt"))


@lru_cache(maxsize=1)
def load_medical_phrases() -> tuple[str, ...]:
    phrases = set(_read_lines(RESOURCES_DIR / "medical_phrases.txt"))
    for values in DOMAIN_TERMS.values():
        phrases.update(values)
    phrases.update(PLACEHOLDER_MAP)
    return tuple(sorted(phrases, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=1)
def load_wordcloud_stopwords() -> frozenset[str]:
    return frozenset(_read_lines(RESOURCES_DIR / "wordcloud_stopwords.txt"))


@lru_cache(maxsize=1)
def load_wordcloud_aliases() -> tuple[tuple[str, str], ...]:
    path = RESOURCES_DIR / "wordcloud_aliases.json"
    if not path.exists():
        return ()
    aliases = json.loads(path.read_text(encoding="utf-8-sig"))
    return tuple(
        sorted(
            ((str(source), str(target)) for source, target in aliases.items()),
            key=lambda item: (-len(item[0]), item[0]),
        )
    )


def _repair_mojibake(text: str) -> str:
    suspicious = sum(text.count(char) for char in "åäæçèéï¼ã€")
    if suspicious < 3:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if len(repaired) >= len(text) * 0.8 else text


def clean_text(text: str) -> str:
    text = _repair_mojibake(str(text or "")).replace("\ufeff", "")
    text = re.sub(r"\s+", "", text)
    return text.replace(";", "；").replace(":", "：").strip()


@lru_cache(maxsize=1)
def load_corpus() -> tuple[CorpusRecord, ...]:
    records: list[CorpusRecord] = []
    for organ, config in ORGAN_CONFIG.items():
        processed_path = PROCESSED_DATA_DIR / config["processed_file"]
        source_path = processed_path if processed_path.exists() else DATA_DIR / config["file"]
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        for split, rows in payload.items():
            for row in rows:
                records.append(
                    CorpusRecord(
                        uid=int(row.get("uid", 0)),
                        organ=organ,
                        organ_name=config["name"],
                        split=str(row.get("split") or split),
                        finding=clean_text(row.get("finding", "")),
                    )
                )
    return tuple(records)


def records_for_organ(organ: str | None = None) -> list[CorpusRecord]:
    records = list(load_corpus())
    if organ in ORGAN_CONFIG:
        return [record for record in records if record.organ == organ]
    return records


@lru_cache(maxsize=1)
def load_base_dictionary() -> dict[str, list[str]]:
    categories = {key: set(values) for key, values in DOMAIN_TERMS.items()}
    technical_path = DATA_DIR / "key_technical_words.txt"
    categories["technical"] = set(_read_lines(technical_path))
    categories["phrase"] = set(_read_lines(RESOURCES_DIR / "medical_phrases.txt"))
    for config in ORGAN_CONFIG.values():
        categories["organ"].update(config["organs"])
    for placeholder, meaning in PLACEHOLDER_MAP.items():
        categories["measurement"].update((placeholder, meaning))
    return {
        key: sorted(values, key=lambda item: (-len(item), item))
        for key, values in categories.items()
    }


@lru_cache(maxsize=1)
def dynamic_dictionary_terms() -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    suffixes = (
        "结节", "回声", "血流信号", "淋巴结", "占位性病变", "条索样改变",
        "不均匀", "清晰", "规整", "增强", "增粗", "胆囊",
    )
    for record in load_corpus():
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", record.finding):
            for size in range(2, min(9, len(segment) + 1)):
                for index in range(len(segment) - size + 1):
                    candidate = segment[index:index + size]
                    if candidate.endswith(suffixes):
                        counts[candidate] += 1
    base = {word for values in load_base_dictionary().values() for word in values}
    terms = [word for word, count in counts.items() if count >= 12 and word not in base]
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=1)
def build_dictionary() -> tuple[str, ...]:
    words = {word for values in load_base_dictionary().values() for word in values}
    words.update(dynamic_dictionary_terms())
    return tuple(sorted(words, key=lambda item: (-len(item), item)))


def write_jieba_userdict(path: Path = JIEBA_USER_DICT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{word} 100000 nz"
        for word in build_dictionary()
        if len(word) > 1 or word in PLACEHOLDER_MAP
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@lru_cache(maxsize=1)
def _init_jieba() -> None:
    if JIEBA_USER_DICT_PATH.exists():
        with JIEBA_USER_DICT_PATH.open("r", encoding="utf-8") as user_dict:
            jieba.load_userdict(user_dict)
    for word in build_dictionary():
        if len(word) > 1 or word in PLACEHOLDER_MAP:
            jieba.add_word(word, freq=100000, tag="nz")


def _clean_tokens(tokens: list[str]) -> list[str]:
    stopwords = load_stopwords()
    return [
        token.strip()
        for token in tokens
        if token.strip()
        and not PUNCTUATION_PATTERN.fullmatch(token)
        and token.strip() not in stopwords
    ]


def _protect_phrases(text: str) -> tuple[str, dict[str, str]]:
    protected = clean_text(text)
    mapping: dict[str, str] = {}
    for index, phrase in enumerate(load_medical_phrases()):
        if phrase not in protected:
            continue
        marker = f"ZXPHRASE{index}Z"
        protected = protected.replace(phrase, marker)
        mapping[marker] = phrase
    return protected, mapping


def jieba_tokenize(text: str) -> list[str]:
    _init_jieba()
    protected, phrase_map = _protect_phrases(text)
    split_pattern = r"(ZXPHRASE\d+Z|_[A-Za-z0-9]+_|[A-Za-z]+)"
    tokens: list[str] = []
    for part in re.split(split_pattern, protected):
        if not part:
            continue
        if part in phrase_map:
            tokens.append(phrase_map[part])
        elif ALNUM_PATTERN.fullmatch(part):
            tokens.append(part)
        else:
            tokens.extend(jieba.lcut(part, HMM=True))
    return _clean_tokens(tokens)


def _scan_alnum(text: str, start: int) -> tuple[str, int]:
    match = ALNUM_PATTERN.match(text, start)
    if match:
        return match.group(0), match.end()
    return text[start], start + 1


def forward_max_match(text: str, max_len: int = 18) -> list[str]:
    text = clean_text(text)
    dictionary = set(build_dictionary())
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if PUNCTUATION_PATTERN.fullmatch(char):
            index += 1
            continue
        if char == "_" or (char.isascii() and char.isalnum()):
            token, index = _scan_alnum(text, index)
            tokens.append(token)
            continue
        match = ""
        for end in range(min(len(text), index + max_len), index, -1):
            candidate = text[index:end]
            if candidate in dictionary:
                match = candidate
                break
        tokens.append(match or char)
        index += len(match) if match else 1
    return _clean_tokens(tokens)


def reverse_max_match(text: str, max_len: int = 18) -> list[str]:
    text = clean_text(text)
    dictionary = set(build_dictionary())
    reversed_tokens: list[str] = []
    index = len(text)
    while index > 0:
        char = text[index - 1]
        if PUNCTUATION_PATTERN.fullmatch(char):
            index -= 1
            continue
        if char == "_" or (char.isascii() and char.isalnum()):
            start = index - 1
            while start > 0 and re.match(r"[A-Za-z0-9_]", text[start - 1]):
                start -= 1
            reversed_tokens.append(text[start:index])
            index = start
            continue
        match = ""
        for start in range(max(0, index - max_len), index):
            candidate = text[start:index]
            if candidate in dictionary:
                match = candidate
                break
        reversed_tokens.append(match or char)
        index -= len(match) if match else 1
    return _clean_tokens(list(reversed(reversed_tokens)))


def tokenize(text: str, mode: str = "jieba") -> list[str]:
    if mode == "forward":
        return forward_max_match(text)
    if mode == "reverse":
        return reverse_max_match(text)
    return jieba_tokenize(text)


POS_NAME_MAP = {
    "n": "名词", "nr": "人名", "ns": "地名", "nt": "机构名", "nz": "专有名词",
    "v": "动词", "a": "形容词", "m": "数词", "q": "量词", "p": "介词",
    "f": "方位词", "x": "非语素字", "eng": "英文",
}


def pos_tag_text(text: str) -> list[dict[str, Any]]:
    _init_jieba()
    dictionary = set(build_dictionary())
    output: list[dict[str, Any]] = []
    for token in jieba_tokenize(text):
        if token in dictionary:
            flag = "nz"
        else:
            tagged = list(pseg.cut(token, HMM=True))
            flag = tagged[0].flag if tagged else "x"
        output.append(
            {
                "word": token,
                "pos": flag,
                "pos_name": POS_NAME_MAP.get(flag, flag),
                "is_medical_term": token in dictionary,
            }
        )
    return output


def token_spans(text: str, tokens: list[str]) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    spans: list[dict[str, Any]] = []
    cursor = 0
    dictionary = set(build_dictionary())
    for token in tokens:
        start = cleaned.find(token, cursor)
        if start < 0:
            start = cleaned.find(token)
        if start < 0:
            continue
        end = start + len(token)
        spans.append(
            {
                "token": token,
                "start": start,
                "end": end,
                "in_dictionary": token in dictionary,
            }
        )
        cursor = end
    return spans


def compare_segmentations(text: str, segmentations: dict[str, list[str]]) -> dict[str, Any]:
    span_map = {
        name: token_spans(text, tokens)
        for name, tokens in segmentations.items()
    }
    boundary_map = {
        name: {span["end"] for span in spans[:-1]}
        for name, spans in span_map.items()
    }
    all_boundaries = set().union(*boundary_map.values()) if boundary_map else set()
    common_boundaries = set.intersection(*boundary_map.values()) if boundary_map else set()
    disagreements = []
    cleaned = clean_text(text)
    for position in sorted(all_boundaries - common_boundaries):
        disagreements.append(
            {
                "position": position,
                "context": cleaned[max(0, position - 8):position + 8],
                "algorithms": [
                    name for name, boundaries in boundary_map.items() if position in boundaries
                ],
            }
        )
    pairwise = {}
    names = list(boundary_map)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            union = boundary_map[left] | boundary_map[right]
            pairwise[f"{left}_vs_{right}"] = round(
                len(boundary_map[left] & boundary_map[right]) / len(union), 4
            ) if union else 1.0
    return {
        "spans": span_map,
        "common_boundary_count": len(common_boundaries),
        "different_boundary_count": len(all_boundaries - common_boundaries),
        "pairwise_agreement": pairwise,
        "disagreements": disagreements,
    }


def analyze_report(text: str, organ: str = "thyroid") -> dict[str, Any]:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    cleaned = clean_text(text)
    segmentations = {
        "jieba": jieba_tokenize(cleaned),
        "forward": forward_max_match(cleaned),
        "reverse": reverse_max_match(cleaned),
    }
    comparison = compare_segmentations(cleaned, segmentations)
    dictionary = set(build_dictionary())
    jieba_tokens = segmentations["jieba"]
    placeholders = [token for token in jieba_tokens if PLACEHOLDER_PATTERN.fullmatch(token)]
    unknown_words = [
        token for token in jieba_tokens
        if len(token) > 1
        and token not in dictionary
        and not PLACEHOLDER_PATTERN.fullmatch(token)
        and not token.isascii()
    ]
    return {
        "organ": organ,
        "organ_name": ORGAN_CONFIG[organ]["name"],
        "original": text,
        "cleaned": cleaned,
        "jieba_tokens": jieba_tokens,
        "forward_tokens": segmentations["forward"],
        "reverse_tokens": segmentations["reverse"],
        "pos_tags": pos_tag_text(cleaned),
        "comparison": comparison,
        "token_stats": {
            "jieba_count": len(jieba_tokens),
            "forward_count": len(segmentations["forward"]),
            "reverse_count": len(segmentations["reverse"]),
            "medical_term_count": sum(token in dictionary for token in jieba_tokens),
            "placeholder_count": len(placeholders),
            "placeholders": placeholders,
            "unknown_words": unknown_words,
        },
        "wordcloud_summary": wordcloud_summary(cleaned),
    }


def _wordcloud_tokens(text: str) -> list[str]:
    normalized = clean_text(text)
    aliases = load_wordcloud_aliases()
    for source, target in aliases:
        normalized = normalized.replace(source, target)
    target_map: dict[str, str] = {}
    targets = sorted({target for _, target in aliases}, key=lambda item: (-len(item), item))
    for index, target in enumerate(targets):
        if target not in normalized:
            continue
        marker = f"WCALIAS{chr(ord('A') + index)}END"
        normalized = normalized.replace(target, marker)
        target_map[marker] = target
    return [target_map.get(token, token) for token in jieba_tokenize(normalized)]


@lru_cache(maxsize=1)
def _wordcloud_term_categories() -> dict[str, str]:
    return {
        term: category
        for category, terms in DOMAIN_TERMS.items()
        for term in terms
    }


def wordcloud_frequencies(text: str, max_words: int = 30) -> dict[str, float]:
    tokens = _wordcloud_tokens(text)
    dictionary = set(build_dictionary())
    alias_targets = {target for _, target in load_wordcloud_aliases()}
    stopwords = load_wordcloud_stopwords()
    actual_counts: Counter[str] = Counter()
    for token in tokens:
        if (
            len(token) <= 1
            or token in stopwords
            or token.isdigit()
            or PLACEHOLDER_PATTERN.fullmatch(token)
        ):
            continue
        if token.isascii() and token not in dictionary:
            continue
        actual_counts[token] += 1
    categories = _wordcloud_term_categories()
    category_boost = {
        "organ": 1.12,
        "lesion": 1.24,
        "echo": 1.16,
        "blood": 1.12,
        "boundary": 1.08,
        "shape": 1.08,
        "status": 1.04,
        "location": 1.02,
    }
    weighted: dict[str, float] = {}
    for token, count in actual_counts.items():
        if token not in dictionary and token not in alias_targets and count < 2:
            continue
        boost = category_boost.get(categories.get(token, ""), 1.08 if token in alias_targets else 1.0)
        if token == "CDFI":
            boost = 0.88
        weighted[token] = round(min(2.6, (1.0 + math.log1p(count)) * boost), 4)
    ranked = sorted(
        weighted.items(),
        key=lambda item: (-item[1], -actual_counts[item[0]], -len(item[0]), item[0]),
    )
    return dict(ranked[:max_words])


def wordcloud_summary(text: str, max_words: int = 30) -> dict[str, Any]:
    weighted = wordcloud_frequencies(text, max_words)
    actual_counts = Counter(_wordcloud_tokens(text))
    ranked = sorted(
        weighted,
        key=lambda token: (-actual_counts[token], -weighted[token], -len(token), token),
    )
    return {
        "word_count": len(weighted),
        "top_words": [
            {"word": token, "count": actual_counts[token]}
            for token in ranked[:6]
        ],
    }


def _wordcloud_font_path() -> Path:
    for path in WORDCLOUD_FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("未找到可用于中文词云的微软雅黑、黑体或宋体字体。")


def _wordcloud_color(
    word: str,
    *_args: Any,
    **_kwargs: Any,
) -> str:
    category = _wordcloud_term_categories().get(word, "")
    category_colors = {
        "organ": "#006d77",
        "lesion": "#9b2c3b",
        "echo": "#245a78",
        "blood": "#0f766e",
        "boundary": "#385170",
        "shape": "#465b73",
        "location": "#32746d",
        "status": "#53657a",
    }
    if category in category_colors:
        return category_colors[category]
    palette = ("#006d77", "#245a78", "#385170", "#7f3545")
    return palette[sum(ord(char) for char in word) % len(palette)]


def generate_wordcloud_png(text: str) -> bytes:
    frequencies = wordcloud_frequencies(text)
    if not frequencies:
        raise ValueError("当前文本过滤后没有可用于生成词云的有效医学词。")
    max_font_size = 72 if len(frequencies) < 8 else 90
    cloud = WordCloud(
        font_path=str(_wordcloud_font_path()),
        width=1200,
        height=360,
        background_color="white",
        max_words=30,
        max_font_size=max_font_size,
        min_font_size=18,
        random_state=42,
        collocations=False,
        prefer_horizontal=1.0,
        relative_scaling=0.35,
        margin=8,
        color_func=_wordcloud_color,
    ).generate_from_frequencies(frequencies)
    output = BytesIO()
    cloud.to_image().save(output, format="PNG")
    return output.getvalue()


def dataset_stats() -> dict[str, Any]:
    records = load_corpus()
    organs: dict[str, Any] = {}
    for organ, config in ORGAN_CONFIG.items():
        subset = [record for record in records if record.organ == organ]
        split_counts = Counter(record.split for record in subset)
        placeholder_counts: Counter[str] = Counter()
        lengths = []
        for record in subset:
            placeholder_counts.update(PLACEHOLDER_PATTERN.findall(record.finding))
            lengths.append(len(record.finding))
        organs[organ] = {
            "name": config["name"],
            "total": len(subset),
            "splits": dict(split_counts),
            "average_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "placeholders": dict(placeholder_counts.most_common()),
        }
    metrics = {}
    if SEGMENTATION_METRICS_PATH.exists():
        metrics = json.loads(SEGMENTATION_METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "total": len(records),
        "organs": organs,
        "dictionary_size": len(build_dictionary()),
        "stopword_count": len(load_stopwords()),
        "phrase_count": len(load_medical_phrases()),
        "placeholder_map": PLACEHOLDER_MAP,
        "segmentation_metrics": metrics,
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
        "finding": record.finding,
        "index": index % len(records),
        "total": len(records),
    }


def dictionary_view() -> dict[str, Any]:
    categories = load_base_dictionary()
    dynamic_terms = list(dynamic_dictionary_terms())
    full_dictionary = list(build_dictionary())
    return {
        "size": len(full_dictionary),
        "categories": categories,
        "summary": {
            "total_terms": len(full_dictionary),
            "dynamic_terms": len(dynamic_terms),
            "stopword_count": len(load_stopwords()),
            "protected_phrase_count": len(load_medical_phrases()),
            "category_count": len(categories),
            "jieba_userdict": str(JIEBA_USER_DICT_PATH),
        },
        "stopwords": sorted(load_stopwords()),
        "protected_phrases": list(load_medical_phrases()),
        "dynamic_examples": dynamic_terms[:120],
    }


def _boundary_set(text: str, tokens: list[str]) -> set[int]:
    spans = token_spans(text, tokens)
    return {span["end"] for span in spans[:-1]}


def segmentation_score(text: str, gold_tokens: list[str], predicted_tokens: list[str]) -> dict[str, float]:
    gold = _boundary_set(text, gold_tokens)
    predicted = _boundary_set(text, predicted_tokens)
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = correct / len(gold) if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct_boundaries": correct,
        "predicted_boundaries": len(predicted),
        "gold_boundaries": len(gold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def load_gold_samples() -> list[dict[str, Any]]:
    return json.loads(GOLD_DATA_PATH.read_text(encoding="utf-8"))


def evaluate_segmentation() -> dict[str, Any]:
    samples = load_gold_samples()
    algorithms = {
        "jieba": jieba_tokenize,
        "forward": forward_max_match,
        "reverse": reverse_max_match,
    }
    totals = {
        name: Counter(correct=0, predicted=0, gold=0)
        for name in algorithms
    }
    organ_totals = {
        organ: {
            name: Counter(correct=0, predicted=0, gold=0)
            for name in algorithms
        }
        for organ in ORGAN_CONFIG
    }
    errors = []
    for sample in samples:
        text = sample["text"]
        gold_tokens = sample["tokens"]
        for name, algorithm in algorithms.items():
            predicted = algorithm(text)
            score = segmentation_score(text, gold_tokens, predicted)
            totals[name].update(
                correct=score["correct_boundaries"],
                predicted=score["predicted_boundaries"],
                gold=score["gold_boundaries"],
            )
            organ_totals[sample["organ"]][name].update(
                correct=score["correct_boundaries"],
                predicted=score["predicted_boundaries"],
                gold=score["gold_boundaries"],
            )
            if predicted != gold_tokens:
                errors.append(
                    {
                        "organ": sample["organ"],
                        "algorithm": name,
                        "text": text,
                        "gold_tokens": gold_tokens,
                        "predicted_tokens": predicted,
                        "f1": score["f1"],
                    }
                )

    def finalize(counter: Counter) -> dict[str, Any]:
        precision = counter["correct"] / counter["predicted"] if counter["predicted"] else 0
        recall = counter["correct"] / counter["gold"] if counter["gold"] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "correct_boundaries": counter["correct"],
            "predicted_boundaries": counter["predicted"],
            "gold_boundaries": counter["gold"],
        }

    return {
        "sample_count": len(samples),
        "overall": {name: finalize(counter) for name, counter in totals.items()},
        "by_organ": {
            organ: {name: finalize(counter) for name, counter in algorithms.items()}
            for organ, algorithms in organ_totals.items()
        },
        "error_examples": sorted(errors, key=lambda item: item["f1"])[:18],
    }
