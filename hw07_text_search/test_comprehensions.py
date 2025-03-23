import unittest

from hw07_text_search.comprehensions import *


class TestComprehensions(unittest.TestCase):

    def setUp(self):
        self.numberList = [-2, -3, 4, 5, 7, 43, 2345, 1343, 13, 40, 24, 221, 5942, 7, 9, 113]
        self.wordList = "All my words are unique and therefore not duplicated".split()
        from hw07_text_search.comprehensions import list, set, dict
        self.assertRaises(NotImplementedError, list)
        self.assertRaises(NotImplementedError, set)
        self.assertRaises(NotImplementedError, dict)

    def test_multiply_by(self): # [1 point]
        expected = [4, 6, -8, -10, -14, -86, -4690, -2686, -26, -80, -48, -442, -11884, -14, -18, -226]
        got = multiply_by(-2, self.numberList)
        self.assertEqual(got, expected)

    def test_get_longest_word(self): # [1 point]
        list_of_sents = ["The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder",
                         "Recurrent models typically factor computation along the symbol positions of the input and output sequences",
                         "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks"
        ]
        expected = get_longest_word(list_of_sents)
        got = ["convolutional", "computation", "transduction"]
        self.assertEqual(got, expected)

    def test_merge_lists(self): # [1 point]
        nested_list = [[-3, 4, 2, 7, -6], [1, 10], [3, 9, 3], [-7]]
        expected = [-3, 4, 2, 7, -6, 1, 10, 3, 9, 3, -7]
        got = merge_lists(nested_list)
        self.assertEqual(got, expected)

    def test_transpose_matrix(self):
        matrix_list = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        got = transpose_matrix(matrix_list)
        self.assertEqual(got, expected)

    def test_map_zip(self): # [1 point]
        expected = {'All': -2, 'my': -3, 'words': 4, 'are': 5, 'unique': 7, 'and': 43, 'therefore': 2345, 'not': 1343,
                    'duplicated': 13}
        got = map_zip(self.wordList, self.numberList)
        expected2 = {-2: 'All', -3: 'my', 4: 'words', 5: 'are', 7: 'unique', 43: 'and', 2345: 'therefore', 1343: 'not',
                     13: 'duplicated'}
        got2 = map_zip(self.numberList, self.wordList)
        self.assertEqual(got, expected)
        self.assertEqual(got2, expected2)

    def test_count_occur(self): # [1 point]
        dict1 = {'antigravity': 5, 'ant': 3, 'antimatter': 10, 'antiquity': 5, 'antonym': 6, 'Anton': 7, 'not': 3, 'antibiotic': 4}
        sum = count_occur(dict1)
        self.assertEqual(sum, 24)
