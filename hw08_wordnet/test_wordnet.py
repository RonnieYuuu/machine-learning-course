from os.path import join, dirname
from unittest import TestCase
from nltk.corpus import wordnet
from src.hw08_wordnet.noun_similarity import get_similarity_scores
from src.hw08_wordnet.find_hyponyms import HyponymSearcher


class WordnetTest(TestCase):

    def setUp(self):
        self.hyponymSearcher = HyponymSearcher(join(dirname(__file__), "ada_lovelace.txt"))

    def test_01_noun_similarity(self): # [4 points]
        pairs = [('car', 'automobile'), ('gem', 'jewel'), ('journey', 'voyage'),
                 ('boy', 'lad'), ('coast', 'shore'), ('asylum', 'madhouse'), ('magician', 'wizard'),
                 ('midday', 'noon'), ('furnace', 'stove'), ('food', 'fruit'), ('bird', 'cock'),
                 ('bird', 'crane'), ('tool', 'implement'), ('brother', 'monk'), ('lad', 'brother'),
                 ('crane', 'implement'), ('journey', 'car'), ('monk', 'oracle'), ('cemetery', 'woodland'),
                 ('food', 'rooster'), ('coast', 'hill'), ('forest', 'graveyard'), ('shore', 'woodland'),
                 ('monk', 'slave'), ('coast', 'forest'), ('lad', 'wizard'), ('chord', 'smile'), ('glass', 'magician'),
                 ('rooster', 'voyage'), ('noon', 'string')]
        results = get_similarity_scores(pairs)
        sim_of_car = [(pair, sim) for pair, sim in results if pair =='car-automobile'][0]
        sim_of_voyage = [(pair, sim) for pair, sim in results if pair =='journey-voyage'][0]
        self.assertEqual(sim_of_car,('car-automobile', 1.0))
        self.assertEqual(sim_of_voyage,('journey-voyage', 0.5))


    def test_02_lemma_names(self): # [6 points]
        self.assertEqual(len(self.hyponymSearcher.noun_lemmas), 1262)


    def test_03_synonyms(self): # [2 points]
        synonyms = self.hyponymSearcher.get_synonyms('car')
        # Example checks, actual values will depend on the context
        self.assertIn('automobile', synonyms)
        self.assertIn('railcar', synonyms)
        self.assertIn('auto', synonyms)

        synonyms_computer = self.hyponymSearcher.get_synonyms('computer')
        self.assertIn('calculator', synonyms_computer)
        self.assertIn('computing_machine', synonyms_computer)

        synonyms_bird = self.hyponymSearcher.get_synonyms('bird')
        self.assertIn('fowl', synonyms_bird)
        self.assertIn('dame', synonyms_bird)

    def test_04_common_synonyms(self): # [2 points]
        common_synonyms = self.hyponymSearcher.get_common_synonyms(['car', 'automobile'])
        self.assertIn('auto', common_synonyms)
        self.assertIn('motorcar', common_synonyms)

        common_synonyms_computer = self.hyponymSearcher.get_common_synonyms(['computer', 'calculator'])
        # Assuming 'computer' and 'calculator' have common synonyms in WordNet
        self.assertIn('estimator', common_synonyms_computer)
        self.assertIn('reckoner', common_synonyms_computer)

    def test_05_find_deepest_common_hypernym(self): # [2 points]
        lemma1, lemma2 = "car", "implement"
        self.assertEqual(self.hyponymSearcher.find_deepest_common_hypernym(lemma1, lemma2), {'instrumentality.n.03'})
        lemma3, lemma4 = "coast", "forest"
        self.assertEqual(self.hyponymSearcher.find_deepest_common_hypernym(lemma3, lemma4), {'object.n.01'})


    def test_05_hypernymOF(self): # [1 point]
        son = wordnet.synsets("son", pos="n")[0]
        relative = wordnet.synsets("relative", pos='n')[0]
        self.assertTrue(self.hyponymSearcher.hypernym_of(son, relative))


    def test_06_hyponyms(self): # [3 points]
        #find words that are hyponyms to the following three synsets
        relative = wordnet.synsets("relative", pos='n')[0]
        science = wordnet.synsets("science", pos='n')[0]
        illness = wordnet.synsets("illness", pos='n')[0]

        hypos_relative = self.hyponymSearcher.get_hyponyms(relative)
        hypos_science = self.hyponymSearcher.get_hyponyms(science)
        hypos_illness = self.hyponymSearcher.get_hyponyms(illness)

        self.assertTrue("father" in hypos_relative)
        self.assertTrue("half-sister" in hypos_relative)
        self.assertTrue("husband" in hypos_relative)

        self.assertTrue("calculus" in hypos_science)
        self.assertTrue("math" in hypos_science)
        self.assertTrue("anatomy" in hypos_science)

        self.assertTrue("cancer" in hypos_illness)
        self.assertTrue("disease" in hypos_illness)
        self.assertTrue("illness" in hypos_illness)
