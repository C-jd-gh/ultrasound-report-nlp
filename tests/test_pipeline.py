import unittest
from io import BytesIO

from PIL import Image

from app import app
from ultrasound_nlp.nlp_pipeline import (
    analyze_report,
    build_dictionary,
    dataset_stats,
    dictionary_view,
    evaluate_segmentation,
    forward_max_match,
    jieba_tokenize,
    load_corpus,
    load_gold_samples,
    load_stopwords,
    reverse_max_match,
    sample_report,
    segmentation_score,
    generate_wordcloud_png,
    wordcloud_frequencies,
)


class SegmentationPipelineTest(unittest.TestCase):
    def test_loads_all_datasets_without_using_labels(self):
        records = load_corpus()
        stats = dataset_stats()
        self.assertEqual(len(records), 7364)
        self.assertEqual(stats["organs"]["thyroid"]["total"], 2457)
        self.assertFalse(hasattr(records[0], "label"))

    def test_medical_phrases_are_kept(self):
        text = (
            "右叶近峡部可见以囊性为主的囊实混合回声结节，"
            "CDFI示周边及内部可探及血流信号。"
        )
        tokens = jieba_tokenize(text)
        self.assertIn("右叶近峡部", tokens)
        self.assertIn("囊性为主的囊实混合回声结节", tokens)
        self.assertIn("周边及内部可探及血流信号", tokens)

    def test_postoperative_and_lymph_phrases_are_kept(self):
        text = "甲状腺全切术后，双侧颈部未见异常肿大淋巴结。"
        tokens = jieba_tokenize(text)
        self.assertIn("甲状腺全切术后", tokens)
        self.assertIn("双侧颈部", tokens)
        self.assertIn("未见异常肿大淋巴结", tokens)

    def test_placeholders_are_atomic(self):
        text = "左叶大小为_3DS_，峡部_SCM_，结节大小约_2DS_。"
        for algorithm in (jieba_tokenize, forward_max_match, reverse_max_match):
            tokens = algorithm(text)
            self.assertIn("_3DS_", tokens)
            self.assertIn("_SCM_", tokens)
            self.assertIn("_2DS_", tokens)

    def test_stopwords_and_punctuation_are_filtered(self):
        tokens = jieba_tokenize("于右叶可见一低回声结节，大小约_2DS_。")
        for stopword in ("于", "可", "见", "一", "大小", "约"):
            self.assertNotIn(stopword, tokens)
        self.assertIn("低回声结节", tokens)

    def test_three_algorithms_return_stable_results(self):
        text = "左叶中下部可见低回声结节，边界清晰。"
        for algorithm in (jieba_tokenize, forward_max_match, reverse_max_match):
            result = algorithm(text)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

    def test_analyze_contract(self):
        result = analyze_report("右叶可见低回声结节，大小约_2DS_。", "thyroid")
        expected = {
            "original", "cleaned", "jieba_tokens", "forward_tokens",
            "reverse_tokens", "pos_tags", "comparison", "token_stats",
        }
        self.assertTrue(expected.issubset(result))
        self.assertIn("pairwise_agreement", result["comparison"])

    def test_segmentation_metric_math(self):
        score = segmentation_score(
            "甲状腺低回声结节",
            ["甲状腺", "低回声结节"],
            ["甲状腺", "低回声结节"],
        )
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["f1"], 1.0)

    def test_gold_dataset_and_evaluation(self):
        samples = load_gold_samples()
        metrics = evaluate_segmentation()
        self.assertEqual(len(samples), 30)
        self.assertEqual(metrics["sample_count"], 30)
        for algorithm in ("jieba", "forward", "reverse"):
            self.assertIn(algorithm, metrics["overall"])
            self.assertGreaterEqual(metrics["overall"][algorithm]["f1"], 0)
            self.assertLessEqual(metrics["overall"][algorithm]["f1"], 1)

    def test_dictionary_contract(self):
        data = dictionary_view()
        self.assertGreater(len(build_dictionary()), 0)
        self.assertGreater(len(load_stopwords()), 0)
        self.assertIn("protected_phrases", data)
        self.assertIn("categories", data)

    def test_sample_does_not_expose_label(self):
        sample = sample_report("mammary", 0)
        self.assertNotIn("label", sample)
        self.assertIn("finding", sample)

    def test_wordcloud_frequency_filter(self):
        frequencies = wordcloud_frequencies(
            "低回声结节，低回声结节，CDFI示未探及血流信号，"
            "大小约_2DS_，周边及内部可探及血流信号。"
        )
        self.assertIn("低回声结节", frequencies)
        self.assertIn("CDFI", frequencies)
        self.assertIn("血流信号", frequencies)
        self.assertNotIn("_2DS_", frequencies)
        self.assertNotIn("约", frequencies)
        self.assertNotIn("周边", frequencies)
        self.assertNotIn("内部", frequencies)
        self.assertNotIn("可见", frequencies)
        self.assertGreater(frequencies["低回声结节"], frequencies["CDFI"])

    def test_wordcloud_aliases_and_word_limit(self):
        frequencies = wordcloud_frequencies(
            "囊性为主的囊实混合回声结节，周边及内部可探及血流信号，"
            "双侧颈部未见明显肿大淋巴结。"
        )
        self.assertIn("囊实混合回声结节", frequencies)
        self.assertIn("血流信号", frequencies)
        self.assertIn("淋巴结未见肿大", frequencies)
        self.assertNotIn("囊性为主的囊实混合回声结节", frequencies)
        self.assertNotIn("未见明显肿大淋巴结", frequencies)
        self.assertLessEqual(len(frequencies), 30)

    def test_wordcloud_png_generation(self):
        image = generate_wordcloud_png("甲状腺右叶可见低回声结节，边界清晰。")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(BytesIO(image)) as rendered:
            self.assertEqual(rendered.size, (1200, 360))

    def test_wordcloud_endpoint(self):
        client = app.test_client()
        response = client.post(
            "/api/wordcloud",
            json={"organ": "thyroid", "text": "甲状腺右叶可见低回声结节。"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "image/png")
        self.assertTrue(response.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_wordcloud_endpoint_rejects_empty_content(self):
        client = app.test_client()
        self.assertEqual(client.post("/api/wordcloud", json={"text": ""}).status_code, 400)
        self.assertEqual(
            client.post("/api/wordcloud", json={"text": "_2DS_，于一。"}).status_code,
            400,
        )

    def test_removed_model_endpoints_return_404(self):
        client = app.test_client()
        for path in (
            "/api/predict",
            "/api/model/metrics",
            "/api/ner/metrics",
            "/api/vector/metrics",
            "/api/cluster",
            "/api/vector/nearest",
            "/api/similar",
        ):
            response = client.get(path)
            self.assertEqual(response.status_code, 404, path)


if __name__ == "__main__":
    unittest.main()
