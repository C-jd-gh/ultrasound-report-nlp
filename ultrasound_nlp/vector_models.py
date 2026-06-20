from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .nlp_pipeline import MODELS_DIR, ORGAN_CONFIG, clean_text, records_for_organ, tokenize


def cluster_model_path(organ: str) -> Path:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    return MODELS_DIR / f"{organ}_kmeans.pkl"


def word2vec_model_path() -> Path:
    return MODELS_DIR / "ultrasound_word2vec.model"


def vector_metrics_path() -> Path:
    return MODELS_DIR / "vector_metrics.json"


def textrank_keywords(text: str, top_k: int = 10, window: int = 4) -> list[dict[str, Any]]:
    tokens = [token for token in tokenize(text) if len(token) > 1]
    if not tokens:
        return []
    graph = nx.Graph()
    graph.add_nodes_from(tokens)
    for index, token in enumerate(tokens):
        for other in tokens[index + 1 : index + window]:
            if token == other:
                continue
            weight = graph[token][other]["weight"] + 1 if graph.has_edge(token, other) else 1
            graph.add_edge(token, other, weight=weight)
    scores = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {token: 1 for token in tokens}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [{"word": word, "score": round(float(score), 4)} for word, score in ranked]


def _documents(organ: str | None = None):
    records = records_for_organ(organ) if organ else list(records_for_organ())
    docs = [record.finding for record in records]
    token_docs = [tokenize(doc) for doc in docs]
    return records, docs, token_docs


def train_word2vec(vector_size: int = 100, window: int = 5, min_count: int = 2) -> dict[str, Any]:
    _, _, token_docs = _documents()
    sentences = [tokens for tokens in token_docs if tokens]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=2,
        sg=1,
        epochs=20,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(word2vec_model_path()))
    return {
        "model_ready": True,
        "model_name": "Word2Vec SkipGram",
        "model_path": str(word2vec_model_path()),
        "vocabulary_size": len(model.wv),
        "vector_size": vector_size,
    }


def nearest_terms(term: str, top_k: int = 10) -> dict[str, Any]:
    path = word2vec_model_path()
    if not path.exists():
        return {
            "model_ready": False,
            "message": "Word2Vec 模型尚未训练，请先运行 python scripts/train_vectors.py",
            "items": [],
        }
    model = Word2Vec.load(str(path))
    if term not in model.wv:
        return {
            "model_ready": True,
            "model_name": "Word2Vec SkipGram",
            "query": term,
            "message": "查询词不在词向量词表中",
            "items": [],
        }
    return {
        "model_ready": True,
        "model_name": "Word2Vec SkipGram",
        "query": term,
        "items": [{"word": word, "score": round(float(score), 4)} for word, score in model.wv.most_similar(term, topn=top_k)],
    }


def _cluster_count(record_count: int) -> int:
    if record_count >= 1200:
        return 6
    if record_count >= 500:
        return 5
    return 4


CLUSTER_PROFILE_RULES = [
    ("术后/复查类", ["术后", "切除", "全切", "原区域", "保乳"]),
    ("淋巴结相关类", ["淋巴结", "淋巴门", "肿大淋巴结", "颈部", "腋下"]),
    ("血流变化类", ["CDFI", "血流信号", "血流丰富", "可探及血流信号"]),
    ("囊实混合结节类", ["囊实混合回声", "囊实混合回声结节", "囊性为主", "实性为主"]),
    ("多发结节类", ["多发结节", "多个结节", "双叶", "双侧", "多个"]),
    ("占位/病灶类", ["占位性病变", "病变", "低回声区", "钙化"]),
    ("弥漫性回声改变类", ["回声增粗", "回声增强", "回声欠均匀", "实质回声", "形态饱满"]),
    ("正常/未见异常类", ["大小形态如常", "未见明确占位性病变", "未见异常血流信号", "回声均匀"]),
    ("胆囊/胆管相关类", ["胆囊", "胆管", "胆囊壁", "扩张"]),
]


def _cluster_profile(top_terms: list[str], representatives: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_text = " ".join(top_terms + [item.get("finding", "") for item in representatives[:2]])
    best_name = "文本模式聚类"
    best_score = 0
    best_hits: list[str] = []
    for name, keywords in CLUSTER_PROFILE_RULES:
        hits = [keyword for keyword in keywords if keyword in evidence_text]
        if len(hits) > best_score:
            best_name = name
            best_score = len(hits)
            best_hits = hits
    summary_terms = "、".join(top_terms[:5]) if top_terms else "暂无高权重词"
    return {
        "name": best_name,
        "summary": f"该簇主要由包含“{summary_terms}”等词的报告组成。",
        "evidence_terms": best_hits[:6],
    }


def train_kmeans_one(organ: str) -> dict[str, Any]:
    records, docs, _ = _documents(organ)
    if not docs:
        raise ValueError(f"No records found for {organ}")
    n_clusters = min(_cluster_count(len(docs)), len(docs))
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        max_features=6000,
        min_df=2,
        ngram_range=(1, 2),
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)

    names = vectorizer.get_feature_names_out()
    clusters = []
    for cluster_id in range(n_clusters):
        indices = [idx for idx, label in enumerate(labels) if int(label) == cluster_id]
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[::-1][:10]
        top_terms = [names[idx] for idx in top_indices if center[idx] > 0]
        distances = cosine_similarity(matrix[indices], center.reshape(1, -1)).ravel() if indices else []
        ranked = sorted(zip(indices, distances), key=lambda item: item[1], reverse=True)[:3]
        representatives = [
            {
                "uid": records[idx].uid,
                "split": records[idx].split,
                "label": records[idx].label,
                "score": round(float(score), 4),
                "finding": records[idx].finding,
            }
            for idx, score in ranked
        ]
        profile = _cluster_profile(top_terms, representatives)
        clusters.append(
            {
                "cluster": cluster_id,
                "profile_name": profile["name"],
                "profile_summary": profile["summary"],
                "profile_evidence": profile["evidence_terms"],
                "count": len(indices),
                "top_terms": top_terms,
                "representatives": representatives,
            }
        )

    score = None
    if n_clusters > 1 and len(set(labels)) > 1:
        sample_size = min(1000, matrix.shape[0])
        score = round(float(silhouette_score(matrix, labels, sample_size=sample_size, random_state=42)), 4)

    bundle = {
        "organ": organ,
        "organ_name": ORGAN_CONFIG[organ]["name"],
        "model_name": "TF-IDF + KMeans",
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "records": [
            {
                "uid": record.uid,
                "split": record.split,
                "label": record.label,
                "finding": record.finding,
            }
            for record in records
        ],
        "clusters": clusters,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with cluster_model_path(organ).open("wb") as fh:
        pickle.dump(bundle, fh)
    return {
        "name": ORGAN_CONFIG[organ]["name"],
        "model_path": str(cluster_model_path(organ)),
        "model_name": "TF-IDF + KMeans",
        "sample_count": len(records),
        "cluster_count": n_clusters,
        "silhouette_score": score,
        "clusters": clusters,
        "label_distribution": dict(Counter(int(label) for label in labels)),
    }


def train_kmeans_all() -> dict[str, Any]:
    payload = {"model_ready": True, "model_name": "TF-IDF + KMeans", "organs": {}}
    for organ in ORGAN_CONFIG:
        payload["organs"][organ] = train_kmeans_one(organ)
    return payload


def train_all_vectors() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_ready": True,
        "word2vec": train_word2vec(),
        "clusters": train_kmeans_all(),
    }
    with vector_metrics_path().open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def vector_metrics() -> dict[str, Any]:
    path = vector_metrics_path()
    if not path.exists():
        return {
            "model_ready": False,
            "message": "向量与聚类模型尚未训练，请先运行 python scripts/train_vectors.py",
            "word2vec": {},
            "clusters": {"organs": {}},
        }
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["model_ready"] = True
    return payload


def cluster_report(text: str, organ: str = "thyroid", top_k: int = 5) -> dict[str, Any]:
    if organ not in ORGAN_CONFIG:
        organ = "thyroid"
    path = cluster_model_path(organ)
    if not path.exists():
        return {
            "model_ready": False,
            "message": "KMeans 聚类模型尚未训练，请先运行 python scripts/train_vectors.py",
            "cluster": None,
            "similar_reports": [],
        }
    cleaned = clean_text(text)
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    vector = bundle["vectorizer"].transform([cleaned])
    cluster = int(bundle["kmeans"].predict(vector)[0])
    cluster_info = next((item for item in bundle["clusters"] if int(item["cluster"]) == cluster), {})
    records = bundle["records"]
    same_cluster_indices = [
        idx
        for idx, label in enumerate(bundle["kmeans"].labels_)
        if int(label) == cluster
    ]
    matrix = bundle["vectorizer"].transform([records[idx]["finding"] for idx in same_cluster_indices])
    scores = cosine_similarity(vector, matrix).ravel()
    ranked = sorted(zip(same_cluster_indices, scores), key=lambda item: item[1], reverse=True)[:top_k]
    return {
        "model_ready": True,
        "model_name": bundle.get("model_name", "TF-IDF + KMeans"),
        "cluster": cluster,
        "profile_name": cluster_info.get("profile_name", "文本模式聚类"),
        "profile_summary": cluster_info.get("profile_summary", ""),
        "cluster_terms": cluster_info.get("top_terms", []),
        "similar_reports": [
            {
                **records[idx],
                "score": round(float(score), 4),
            }
            for idx, score in ranked
        ],
    }
