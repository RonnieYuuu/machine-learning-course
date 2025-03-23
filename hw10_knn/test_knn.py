from unittest import TestCase

from classification import DocumentCollection, KNNClassifier, TextDocument


dir_train = "../data/20news-bydate/20news-bydate-train/"
dir_test = "../data/20news-bydate/20news-bydate-test/"


class ClassificationTest(TestCase):

    @classmethod
    def setUpClass(cls):
        doc_collection_train = DocumentCollection.from_dir(dir_train)
        cls.classifier = KNNClassifier(n_neighbors=4)
        cls.classifier.fit(doc_collection_train)
        vectorsOfTrainDocs = cls.classifier.vectorsOfDoc_collection
        
        test_doc = TextDocument.from_file(dir_test+'alt.atheism/53068', 'alt.atheism')
        vecTestDoc = doc_collection_train.tfidf(test_doc.token_counts)
        #2.1
        cls.dist = cls.classifier.calculate_similarities(vecTestDoc, vectorsOfTrainDocs)
        #2.2
        cls.ordered = cls.classifier.order_nearest_to_farthest(cls.dist)
        #2.3
        cls.k_nearest_labels = cls.classifier.labels_k_closest(cls.ordered)
        #2.4
        cls.label = cls.classifier.choose_one(cls.k_nearest_labels)

    def test_01_calc_sims(self): # [1 point]
        sorted_dist = sorted(self.dist)
        self.assertEqual(sorted_dist[-1][1], 'alt.atheism')

    def test_02_order_near_to_far(self): # [1 point]
        test_distances=[(0.2,"c"), (0.5,"b"), (0.7,"a")]
        self.assertEqual([(0.7,"a"), (0.5,"b"), (0.2,"c")], self.classifier.order_nearest_to_farthest(test_distances))
        self.assertEqual(self.ordered[0][1], 'alt.atheism')

    def test_03_filter_by_similarity_threshold(self): # [1 point]
        test_similarities = [(0.9, "cat1"), (0.7, "cat2"), (0.4, "cat3"), (0.2, "cat4")]
        threshold = 0.5
        filtered = self.classifier.filter_by_similarity_threshold(test_similarities, threshold)
        self.assertEqual(len(filtered), 2)
        self.assertNotIn((0.4, "cat3"), filtered)
        self.assertNotIn((0.2, "cat4"), filtered)

    def test_04_labels_k_closest(self): # [1 point]
        self.assertEqual(self.k_nearest_labels[0], 'alt.atheism')

    def test_05_choose_neighbor(self): # [1 point]
        winner = self.classifier.choose_one(['rec.sport.hockey', 'rec.sport.baseball', 'rec.motorcycles', 'rec.sport.hockey'])
        self.assertEqual(winner, 'rec.sport.hockey')
        self.assertEqual(self.label, 'alt.atheism')

    def test_06_classify(self): # [1 point]
        test_file = dir_test+'alt.atheism/53068'
        self.assertEqual(self.classifier.classify(test_file), 'alt.atheism')

    def test_07_accuracy(self): # [2 point]
        test_files = [
            (dir_test + 'alt.atheism/53272', 'alt.atheism'),
            (dir_test + 'sci.med/59225', 'sci.med'),
            (dir_test + 'comp.graphics/38758', 'comp.graphics'),
            (dir_test + 'rec.autos/103007', 'rec.autos')
        ]

        predicted_labels = [self.classifier.classify(file) for file, _ in test_files]
        gold_labels = [cat for _, cat in test_files]

        result = self.classifier.get_accuracy(gold_labels, predicted_labels)

        expected_output = {
            "overall_accuracy": 75,
            "category_accuracies": {
                "alt.atheism": 0,
                "sci.med": 100,
                "comp.graphics": 100,
                "rec.autos": 100
            }
        }

        self.assertEqual(result["overall_accuracy"], expected_output["overall_accuracy"])

        self.assertDictEqual(result["category_accuracies"], expected_output["category_accuracies"])

