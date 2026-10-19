import unittest

from retrieval.metrics import strict_doc_at_k, strict_page_at_k


class StrictMetricTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "evidence_sets": [
                [{"file": "doc", "pages": [1, 2]}],
                [{"file": "doc", "pages": [3]}],
            ]
        }

    def test_complete_original_gold(self):
        recall = [{"file": "doc", "page": 1}, {"file": "doc", "page": 2}]
        self.assertTrue(strict_page_at_k(self.item, recall, 10, "gold"))

    def test_complete_equivalent_combination(self):
        recall = [{"file": "doc", "page": 3}]
        self.assertFalse(strict_page_at_k(self.item, recall, 10, "gold"))
        self.assertTrue(strict_page_at_k(self.item, recall, 10, "equivalent"))

    def test_partial_combination_is_failure(self):
        recall = [{"file": "doc", "page": 1}]
        self.assertTrue(strict_doc_at_k(self.item, recall, 10, "gold"))
        self.assertFalse(strict_page_at_k(self.item, recall, 10, "gold"))

    def test_combinations_are_not_merged(self):
        item = {"evidence_sets": [
            [{"file": "a", "pages": [1]}, {"file": "b", "pages": [1]}],
            [{"file": "a", "pages": [2]}, {"file": "b", "pages": [2]}],
        ]}
        mixed = [{"file": "a", "page": 1}, {"file": "b", "page": 2}]
        self.assertFalse(strict_page_at_k(item, mixed, 10, "equivalent"))

    def test_duplicates_do_not_change_top_k(self):
        recall = [{"file": "doc", "page": 1}, {"file": "doc", "page": 1}, {"file": "doc", "page": 2}]
        self.assertTrue(strict_page_at_k(self.item, recall, 2, "gold"))


if __name__ == "__main__":
    unittest.main()
