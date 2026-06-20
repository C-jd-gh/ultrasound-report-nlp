from __future__ import annotations

import re
from typing import Any

from .nlp_pipeline import DOMAIN_TERMS, REGEX_RULES, clean_text, extract_regex_matches


def _sentences(text: str) -> list[tuple[int, str]]:
    cleaned = clean_text(text)
    parts: list[tuple[int, str]] = []
    start = 0
    for match in re.finditer(r"[。；;]", cleaned):
        sentence = cleaned[start : match.start()]
        if sentence:
            parts.append((start, sentence))
        start = match.end()
    tail = cleaned[start:]
    if tail:
        parts.append((start, tail))
    return parts


def _first_in_sentence(sentence: str, terms: list[str]) -> str | None:
    for term in sorted(terms, key=len, reverse=True):
        if term in sentence:
            return term
    return None


def _first_regex(sentence: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            return match.group(0)
    return None


def _organ_location(sentence: str) -> str | None:
    location = _first_regex(sentence, REGEX_RULES["location"])
    organ = _first_in_sentence(
        sentence,
        DOMAIN_TERMS["organ"] + ["甲状腺", "乳腺", "肝脏", "胆囊", "胰腺", "脾脏"],
    )
    if organ and location and organ not in location:
        return f"{organ}{location}"
    return location or organ


def _trigger(sentence: str) -> str | None:
    for word in ["可见", "见", "探及", "显示", "示", "未见"]:
        if word in sentence:
            return word
    return None


def extract_relations(text: str, organ: str = "thyroid") -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    regex_matches = extract_regex_matches(text)

    for offset, sentence in _sentences(text):
        lesion = _first_in_sentence(sentence, DOMAIN_TERMS["lesion"] + ["结节", "囊肿", "钙化", "占位性病变"])
        if not lesion:
            continue
        subject = _organ_location(sentence) or "部位未明确"
        size = _first_regex(sentence, REGEX_RULES["measurement"])
        echo = _first_regex(sentence, REGEX_RULES["echo"])
        boundary = _first_regex(sentence, REGEX_RULES["boundary"])
        shape = _first_regex(sentence, REGEX_RULES["shape"])
        blood = _first_regex(sentence, REGEX_RULES["blood"])
        related_rules = [
            item
            for item in regex_matches
            if offset <= int(item["start"]) and int(item["end"]) <= offset + len(sentence)
        ]
        relations.append(
            {
                "subject": subject,
                "predicate": _trigger(sentence) or "描述",
                "object": lesion,
                "size": size,
                "echo": echo,
                "boundary": boundary,
                "shape": shape,
                "blood": blood,
                "sentence": sentence,
                "evidence": f"{subject} -> {lesion}",
                "rule": "句子切分 + 触发词 + 最近部位/属性匹配",
                "start": offset,
                "end": offset + len(sentence),
                "regex_rule_count": len(related_rules),
            }
        )

    return relations
