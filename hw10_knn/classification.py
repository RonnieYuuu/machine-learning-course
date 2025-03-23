from nltk import FreqDist, word_tokenize
from collections import defaultdict
import os, math
from collections import Counter


def dot(dict_a, dict_b):
    return sum([dict_a.get(tok) * dict_b.get(tok, 0) for tok in dict_a])


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


class TextDocument:
    def __init__(self, text, id=None, category=None):
        self.text = text
        self.token_counts = FreqDist(normalized_tokens(text))
        self.id = id
        self.category = category

    @classmethod
    def from_file(cls, filename, category):
        with open(filename, 'r', encoding="ISO-8859-1") as myfile:
            text=myfile.read().strip()
        return cls(text, filename, category)


class DocumentCollection:
    def __init__(self, term_to_df, term_to_docids, docid_to_doc,doc_to_category):
        # string to int
        self.term_to_df = term_to_df
        # string to set of string
        self.term_to_docids = term_to_docids
        # string to TextDocument
        self.docid_to_doc = docid_to_doc
        # TextDocument to category
        self.doc_to_category = doc_to_category

    @classmethod
    def from_dir(cls, dir):
        files = [(os.path.join(root, name), os.path.relpath(root, dir)) for root, dirs, f in os.walk(dir, topdown=False) for name in f]
        docs = [TextDocument.from_file(f, cat) for f, cat in files]
        return cls.from_document_list(docs)

    @classmethod
    def from_document_list(cls, docs):
        term_to_df = defaultdict(int)
        term_to_docids = defaultdict(set)
        docid_to_doc = dict()
        doc_to_category = dict()
        for doc in docs:
            docid_to_doc[doc.id] = doc
            doc_to_category[doc] = doc.category
            for token in doc.token_counts.keys():
                term_to_df[token] += 1
                term_to_docids[token].add(doc.id)
        return cls(term_to_df, term_to_docids, docid_to_doc, doc_to_category)

    def tfidf(self, counts):
        N = len(self.docid_to_doc)
        return {tok: tf * math.log(N/self.term_to_df[tok]) for tok,tf in counts.items() if tok in self.term_to_df}

    def cosine_similarity(self, weighted_a, weighted_b):
        dot_ab = dot(weighted_a, weighted_b)
        norm_a = math.sqrt(dot(weighted_a, weighted_a))
        norm_b = math.sqrt(dot(weighted_b, weighted_b))
        if norm_a == 0 or norm_b == 0:
            return 0
        else:
            return dot_ab / (norm_a * norm_b)


class KNNClassifier:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.doc_collection = None
        self.vectorsOfDoc_collection = None

    def fit(self, doc_collection):
        self.doc_collection = doc_collection
        self.vectorsOfDoc_collection = [(doc, self.doc_collection.tfidf(doc.token_counts))
                                        for doc in self.doc_collection.docid_to_doc.values()]

    def calculate_similarities(self, vec_test_doc, vectors_of_train_docs):
        #TODO calculate similarities between test and train documents and label them [(similarity, label),...]
        # have a look at the unittest to see what is the type of vectorsOfTrainDocs
        similarities = []
        for train_doc, vec_train_doc in vectors_of_train_docs:
            similarity = self.doc_collection.cosine_similarity(vec_test_doc, vec_train_doc)
            similarities.append((similarity, self.doc_collection.doc_to_category[train_doc]))
        return similarities

    def order_nearest_to_farthest(self, similarities):
        #TODO order the labeled points from nearest to farthest
        return sorted(similarities, reverse=True, key=lambda x: x[0])

    def filter_by_similarity_threshold(self, similarities, threshold):
        # TODO filter out the (similarity, label) pairs that have similarities
        #  below the specified threshold and return the filtered list.
        return [(sim, label) for sim, label in similarities if sim >= threshold]

    def labels_k_closest(self, sorted_similarities):
        #TODO find the labels for the k closest points
        return [label for _, label in sorted_similarities[:self.n_neighbors]]

    def choose_one(self, labels) :
        #TODO reduce k until you find a unique winner
        label_counts = Counter(labels)
        # find the most common label
        most_common = label_counts.most_common()
        if len(most_common) == 1:
            return most_common[0][0]
        # if multiple labels are the same common, pop the label list items based on self.n_neighbors(from end of list)
        while len(most_common) > 1 and self.n_neighbors > 1:
            self.n_neighbors -= 1
            most_common = label_counts.most_common(self.n_neighbors)
        return most_common[0][0]

    def classify(self, test_file):
        #TODO classify test document, have a look at the slides of the course for some help with the order of commands
        test_doc = TextDocument.from_file(test_file, 'unknowcat')
        vec_test_doc = self.doc_collection.tfidf(test_doc.token_counts)

        # calculate similarity
        similarities = self.calculate_similarities(vec_test_doc, self.vectorsOfDoc_collection)

        # similarities sorted from nearest to furthest
        sorted_similarities = self.order_nearest_to_farthest(similarities)

        # labels of nearest k doc
        k_labels = self.labels_k_closest(sorted_similarities)

        # find the nearest one
        return self.choose_one(k_labels)

    def get_accuracy(self, gold, predicted):
        # TODO: Implement a method to calculate and return both the overall accuracy and per-category accuracies.
        # The output should be a dictionary with:
        # 1. "overall_accuracy": The percentage of correctly classified test documents,
        #    rounded to the nearest integer.
        # 2. "category_accuracies": A dictionary with category labels as keys and
        #    their respective accuracies as percentages. Only include categories
        #    present in the test set.
        #
        # Example return format:
        # {"overall_accuracy": 75,
        #  "category_accuracies": {"alt.atheism": 0, "sci.med": 100, "comp.graphics": 100, "rec.autos": 100} }
        correct = 0
        category_counts = defaultdict(int)
        category_correct = defaultdict(int)

        for g, p in zip(gold, predicted):
            category_counts[g] += 1
            if g == p:
                correct += 1
                category_correct[g] += 1

        overall_accuracy = round((correct / len(gold)) * 100)

        category_accuracies = {cat: round((category_correct[cat] / category_counts[cat]) * 100)
                               for cat in category_counts}

        return {"overall_accuracy": overall_accuracy, "category_accuracies": category_accuracies}

