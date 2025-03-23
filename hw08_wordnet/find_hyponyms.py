import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class HyponymSearcher(object):
    def __init__(self, text_path):

        self.noun_lemmas = []

        #TODO Read text as a string
        with open (text_path, 'r', encoding='utf-8') as file:
            text = file.read()

        #TODO Split into sentences: use nltk.sent_tokenize
            sentences = nltk.sent_tokenize(text)

        #TODO Split into tokens: use nltk.word_tokenize
            for sent in sentences:
                tokens = [word for word in nltk.word_tokenize(sent)]

        #TODO Perform POS tagging on all tokens of all sentences (not on each sentence separately)
            tagged_tokens = [token for token in nltk.pos_tag(nltk.word_tokenize(text))]

        #TODO lemmatize nouns (any token whose POS tags starts with "N"): use WordNetLemmatizer()
            wnl = WordNetLemmatizer()
            lemmatized = [wnl.lemmatize(tup[0]) for tup in tagged_tokens if tup[1].startswith("N")]

        #TODO determine all noun lemmas and save it in self.noun_lemmas
            self.noun_lemmas = lemmatized

    def get_synonyms(self, lemma, k=None):
        #TODO Returns a list of synonyms for the given lemma
        synsets = wn.synsets(lemma)
        synonyms = set()
        for synset in synsets:
            for synonym in synset.lemmas():
                if synonym.name() != lemma:
                    synonyms.add((synonym.name(), synset.max_depth()))
        sorted_synonyms = sorted(synonyms, key=lambda x: x[1], reverse=True)
        synonym_names = [name for name, depth in sorted_synonyms]
        return synonym_names[:k] if k else synonym_names

    def get_common_synonyms(self, lemmas):
        #TODO Returns a list of common synonyms for the given list of lemmas (like ['car', 'automobile'])
        synonym_sets = [set(self.get_synonyms(lemma)) for lemma in lemmas]
        common_synonyms = set.intersection(*synonym_sets)
        return list(common_synonyms)

    def find_deepest_common_hypernym(self, lemma1, lemma2):
        # TODO return a set of names of the hypernyms with the greatest max_depth in the WordNet hierarchy
        #  among all lowest common hypernyms shared by any pair of synsets of the two input words
        lch = []
        for st1 in wn.synsets(lemma1):
            for st2 in wn.synsets(lemma2):
                lch.extend(st1.lowest_common_hypernyms(st2))
        deepest = max((synset.max_depth() for synset in lch), default=0)
        dch = {synset.name() for synset in lch if synset.max_depth() == deepest}
        return dch

    def hypernym_of(self,synset1, synset2):
        #TODO Is synset2 a hypernym of synset 1? (Or the same synset), return True or False

        if synset1 == synset2:
            return True

        for hypernym in synset1.hypernyms():
            if self.hypernym_of(hypernym, synset2):
                return True

        return False

    def get_hyponyms(self,hypernym):
        #TODO determine set of noun lemmas in ada_lovelace.txt that are hyponyms of the given hypernym
        # use the implemented method hypernymOf(self, synset1, synset2)
        hypos = []

        for lemma in self.noun_lemmas:
            synsets = wn.synsets(str(lemma))
            for synset in synsets:
                if HyponymSearcher.hypernym_of(self, synset, hypernym) == 1:
                    hypos.append(lemma)

        return hypos
