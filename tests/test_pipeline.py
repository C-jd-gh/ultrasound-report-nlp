import unittest

from nlp_pipeline import (
    analyze_report,
    dataset_stats,
    extract_entities,
    extract_regex_matches,
    load_corpus,
    normalize_text,
    sample_report,
    tokenize,
)


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
        self.assertIn("template_report", result)
        self.assertGreater(len(result["tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
