import os
from unittest import TestCase
from src.hw09_kmeans.kmeans import Reader
from src.hw09_kmeans.kmeans import Kmeans

filename = os.path.join(os.path.dirname(__file__), "courses.txt")


class ClusteringTest(TestCase):

    def setUp(self):
        self.reader = Reader(filename)
        self.clusterer = Kmeans(3)

    def test_01_courses(self):  # [1 point]
        courses = self.reader.courses  # returns list of courses
        self.assertEqual(courses[:3], ['Bioinformatik', 'Didaktik der Informatik', 'Didaktik der Mathematik'])
        self.assertIn('Medieninformatik', courses)  # Check if another course is in the list
        self.assertNotIn('Astrophysics', courses)  # Check for a course that shouldn't be in the list

    def test_02_normalize(self):  # [1 point]
        word1 = "(Studienrichtung"
        normalized_word1 = self.reader.normalize_word(word1)  # returns list of courses
        self.assertEqual(normalized_word1, "studienrichtung")

        word2 = "Computer-Science!"
        normalized_word2 = self.reader.normalize_word(word2)
        self.assertEqual(normalized_word2, "computerscience")

        word3 = "Data@Analysis"
        normalized_word3 = self.reader.normalize_word(word3)
        self.assertEqual(normalized_word3, "dataanalysis")

    def test_03_vocabulary(self):  # [1 point]
        words = self.reader.vocabulary
        self.assertEqual(words[:3], ['albanologie', 'allgemeine', 'als'])
        self.assertIn('informatik', words)  # Check if a specific word is in vocabulary
        self.assertNotIn('quantum', words)  # Check for a word that shouldn't be in the vocabulary

    def test_04_distance(self):  # [1 point]
        a = [1, 2, 3]
        b = [4, 5, 6]
        euclidean_dist = self.clusterer.euclidian_distance(a, b)
        self.assertEqual(int(euclidean_dist), 5)

        c = [7, 8, 9]
        d = [10, 11, 12]
        euclidean_dist_cd = self.clusterer.euclidian_distance(c, d)
        self.assertEqual(int(euclidean_dist_cd), 5)

    def test_05_vector_mean(self):  # [1 point]
        vectors = [[1, 2, 3], [4, 5, 6]]
        mean = self.clusterer.vector_mean(vectors)
        self.assertEqual(mean, [2.5, 3.5, 4.5])

        vectors2 = [[7, 8, 9], [10, 11, 12]]
        mean2 = self.clusterer.vector_mean(vectors2)
        self.assertEqual(mean2, [8.5, 9.5, 10.5])

    def test_06_classify(self):  # [2 point]
        vectorspaced_data = self.reader.vector_spaced_data

        # Train the model
        self.clusterer.train(vectorspaced_data)
        clusters = [self.clusterer.classify(vec) for vec in vectorspaced_data]

        # Check that each input vector has been assigned a cluster
        self.assertEqual(len(clusters), len(vectorspaced_data))

        for i, vec in enumerate(vectorspaced_data):
            # Get the cluster index for each vector
            closest_cluster = clusters[i]

            # Check if this is indeed the closest center
            distances = [self.clusterer.euclidian_distance(vec, mean) for mean in self.clusterer.means]
            min_distance = min(distances)

            # Verify that the classify method has selected the cluster center with the minimum distance
            self.assertEqual(min_distance, distances[closest_cluster])