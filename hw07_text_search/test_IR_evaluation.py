import unittest
from hw07_text_search.IR_evaluation import calculate_precision_at_k, calculate_r_precision

class TestIREvaluation(unittest.TestCase):

    def setUp(self):
        self.ranked_docs = ["doc1.txt", "doc2.txt", "doc3.pdf", "report1.docx", "summary.pdf"]
        self.relevant_docs = ["doc3.pdf", "report1.docx", "summary.pdf", "analysis.txt", "overview.docx"]

    def test_calculate_precision_at_k(self): # [2 points]
        expected_precision_at_3 = 1 / 3  # 1 relevant documents in top 3 ranked
        result_precision_at_3 = calculate_precision_at_k(self.ranked_docs, self.relevant_docs, 3)
        self.assertEqual(result_precision_at_3, expected_precision_at_3)

    def test_calculate_r_precision(self): # [1 point]
        num_relevant = len(self.relevant_docs)
        expected_r_precision = 3 / num_relevant  # 3 relevant documents in top 5 ranked
        result_r_precision = calculate_r_precision(self.ranked_docs, self.relevant_docs)
        self.assertEqual(result_r_precision, expected_r_precision)

if __name__ == '__main__':
    unittest.main()
