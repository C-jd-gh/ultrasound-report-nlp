import unittest

from ultrasound_nlp.nlp_pipeline import (
    analyze_report,
    cluster_for_report,
    dataset_stats,
    extract_entities,
    extract_regex_matches,
    load_corpus,
    model_metrics,
    nearest_vector_terms,
    ner_metrics,
    normalize_text,
    pos_tag_text,
    predict_label,
    risk_tendency,
    sample_report,
    tokenize,
    vector_model_metrics,
)
from ultrasound_nlp.relation_extractor import extract_relations
from ultrasound_nlp.vector_models import textrank_keywords


class PipelineTest(unittest.TestCase):
    def test_loads_all_datasets(self):
        records = load_corpus()
        stats = dataset_stats()
        self.assertGreater(len(records), 0)
        self.assertIn("thyroid", stats["organs"])
        self.assertIn("mammary", stats["organs"])
        self.assertIn("liver", stats["organs"])
        self.assertGreater(stats["organs"]["thyroid"]["splits"]["train"], 0)

    def test_segmentation_keeps_medical_terms(self):
        text = "右叶可见一低回声结节，边界清晰，形态规整，CDFI示未探及血流信号。"
        tokens = tokenize(text)
        self.assertIn("低回声结节", tokens)
        self.assertIn("边界清晰", tokens)
        self.assertIn("CDFI", tokens)
        self.assertNotIn("CD", tokens)
        self.assertNotIn("FI示腺体内", tokens)

    def test_jieba_mode_keeps_medical_terms(self):
        text = "甲状腺全切术后，原区域未见明确占位性病变。双侧颈部未见异常肿大淋巴结。"
        tokens = tokenize(text, mode="jieba")
        self.assertIn("甲状腺全切术后", tokens)
        self.assertIn("原区域", tokens)
        self.assertIn("未见明确占位性病变", tokens)
        self.assertIn("双侧颈部", tokens)
        self.assertIn("未见异常肿大淋巴结", tokens)

    def test_segmentation_avoids_noisy_dynamic_words(self):
        text = "甲状腺大小形态如常，于左叶腺体内可见多发结节，大者位于上极，呈囊实混合回声，大小约_2DS_。"
        tokens = tokenize(text)
        self.assertIn("甲状腺", tokens)
        self.assertIn("大小形态如常", tokens)
        self.assertIn("多发结节", tokens)
        self.assertIn("囊实混合回声", tokens)
        self.assertIn("_2DS_", tokens)
        self.assertNotIn("左叶腺体内可", tokens)
        self.assertNotIn("大", tokens)
        self.assertNotIn("小", tokens)
        self.assertNotIn("者", tokens)
        self.assertNotIn("位", tokens)

    def test_segmentation_keeps_negation_phrases(self):
        text = "未见明确占位性病变。双侧颈部未见明显肿大淋巴结。"
        tokens = tokenize(text)
        self.assertIn("未见明确占位性病变", tokens)
        self.assertIn("未见明显肿大淋巴结", tokens)

    def test_segmentation_handles_postoperative_report(self):
        text = "甲状腺全切术后，原区域未见明确占位性病变。双侧颈部未见异常肿大淋巴结。"
        tokens = tokenize(text)
        self.assertIn("甲状腺全切术后", tokens)
        self.assertIn("原区域", tokens)
        self.assertIn("未见明确占位性病变", tokens)
        self.assertIn("双侧颈部", tokens)
        self.assertIn("未见异常肿大淋巴结", tokens)
        self.assertNotIn("原", tokens)
        self.assertNotIn("区", tokens)
        self.assertNotIn("域", tokens)
        self.assertNotIn("结", tokens)

    def test_segmentation_handles_diffuse_thyroid_report(self):
        text = "甲状腺体积正常，形态欠规则，腺体内回声增粗、增强，不均匀，部分可见条索样改变，腺体内未见明确占位性病变，CDFI示腺体内血流信号未见明显异常。"
        tokens = tokenize(text)
        self.assertIn("甲状腺体积正常", tokens)
        self.assertIn("形态欠规则", tokens)
        self.assertIn("腺体内回声增粗", tokens)
        self.assertIn("增强", tokens)
        self.assertIn("不均匀", tokens)
        self.assertIn("部分可见条索样改变", tokens)
        self.assertIn("未见明确占位性病变", tokens)
        self.assertIn("CDFI", tokens)
        self.assertIn("腺体内血流信号", tokens)
        self.assertIn("未见明显异常", tokens)
        for noisy in ["正", "常", "不", "均", "匀", "条", "索", "样", "改", "变", "显", "异"]:
            self.assertNotIn(noisy, tokens)

    def test_regex_extracts_core_fields(self):
        text = "左叶中部可见一低回声结节，大小约_2DS_，边界清晰，CDFI示可探及血流信号。"
        matches = extract_regex_matches(text)
        values = [item["value"] for item in matches]
        self.assertIn("_2DS_", values)
        self.assertTrue(any("左叶" in value for value in values))
        self.assertTrue(any("CDFI" in value for value in values))

    def test_entities_include_lesion_event(self):
        text = "左叶中部可见一低回声结节，大小约_2DS_，边界清晰，形态规整，CDFI示未探及血流信号。"
        entities = extract_entities(text, "thyroid")
        lesion_events = [item for item in entities if item["type"] == "lesion_event"]
        self.assertGreaterEqual(len(lesion_events), 1)
        self.assertEqual(lesion_events[0]["attributes"]["size"], "_2DS_")

    def test_normalization_and_template(self):
        text = "边界清楚，形态规则，CDFI示未探及血流信号。"
        normalized = normalize_text(text)
        self.assertIn("边界清晰", normalized)
        self.assertIn("形态规整", normalized)
        self.assertIn("未见血流信号", normalized)

    def test_analyze_sample_report(self):
        sample = sample_report("liver")
        result = analyze_report(sample["finding"], "liver")
        self.assertIn("tokens", result)
        self.assertIn("pos_tags", result)
        self.assertIn("keywords_textrank", result)
        self.assertIn("sequence_entities", result)
        self.assertIn("relations", result)
        self.assertIn("risk_tendency", result)
        self.assertIn("cluster", result)
        self.assertIn("prediction", result)
        self.assertIn("template_report", result)
        self.assertGreater(len(result["tokens"]), 0)

    def test_pos_tags_contract(self):
        tags = pos_tag_text("甲状腺右叶见低回声结节，边界清晰。")
        self.assertGreater(len(tags), 0)
        self.assertIn("word", tags[0])
        self.assertIn("pos", tags[0])
        self.assertIn("is_medical_term", tags[0])

    def test_textrank_keywords_contract(self):
        keywords = textrank_keywords("甲状腺右叶见低回声结节，边界清晰，形态规整。")
        self.assertGreater(len(keywords), 0)
        self.assertIn("word", keywords[0])
        self.assertIn("score", keywords[0])

    def test_relation_extraction_contract(self):
        relations = extract_relations("甲状腺右叶见低回声结节，大小约_2DS_，CDFI示未探及血流信号。")
        self.assertGreaterEqual(len(relations), 1)
        self.assertIn("subject", relations[0])
        self.assertIn("object", relations[0])

    def test_risk_tendency_contract(self):
        result = risk_tendency("结节形态欠规整，边界欠清晰，可见微小钙化。")
        self.assertIn("level", result)
        self.assertIn("suspicious_evidence", result)
        self.assertGreater(len(result["suspicious_evidence"]), 0)

    def test_prediction_contract(self):
        result = predict_label("甲状腺大小形态如常，未见明确占位性病变。", "thyroid")
        self.assertIn("model_ready", result)
        self.assertIn("predicted_label", result)
        self.assertIn("top_labels", result)

    def test_model_metrics_contract(self):
        result = model_metrics()
        self.assertIn("model_ready", result)
        self.assertIn("organs", result)

    def test_new_model_metric_contracts(self):
        self.assertIn("model_ready", ner_metrics())
        self.assertIn("model_ready", vector_model_metrics())

    def test_cluster_and_vector_contracts(self):
        cluster = cluster_for_report("甲状腺右叶见低回声结节。", "thyroid")
        nearest = nearest_vector_terms("结节")
        self.assertIn("model_ready", cluster)
        self.assertIn("model_ready", nearest)


if __name__ == "__main__":
    unittest.main()
