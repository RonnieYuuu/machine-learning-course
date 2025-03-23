import string
import numpy as np
import random


class Reader:

    def __init__(self, path):
        self.path = path
        self.punctuation = set(string.punctuation)
        self.courses = self.get_lines()
        self.vocabulary = self.get_vocabulary()
        self.vector_spaced_data = self.data_to_vectorspace()

    def get_lines(self):
        #TODO return list of courses from file
        with open(self.path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
            return lines

    def normalize_word(self,word):
        #TODO normalize word by lower casing and deleting punctuation from word
        #TODO use string of punctuation symbols (self.punctuation)
        normalized_word = ''.join(char for char in word if char not in self.punctuation).lower()
        return normalized_word

    def get_vocabulary(self):
        #TODO return list of unique, normalized words from file and sort them alphabetically
        #TODO to normalize the words, use the previously implemented method normalize_words(self,word)
        unique_words = set()
        normalized_words = []
        for line in self.courses:
            words = line.split()
            for word in words:
                normalized_words.append(self.normalize_word(word))
        unique_words.update(normalized_words)
        return sorted(unique_words)

    def vectorspaced(self,course):
        """ converts the given course, which is a string, to a one-hot vector,
        i.e., a vector filled with 0s, except for those positions associated with the
        words of the given course in the vocabulary. These positions are filled with 1."""
        course_components = [self.normalize_word(word) for word in course.split()]
        vectors = [int(word in course_components) for word in self.vocabulary]
        return vectors

    def data_to_vectorspace(self):
        """ convert all courses of the Reader to one-hot-vectors"""
        return [self.vectorspaced(course) for course in self.courses if course]


class Kmeans:
    """performs k-means clustering"""

    def __init__(self, k):
        self.k = k
        self.means = None

    def euclidian_distance(self, x,y):
        #TODO calculate Euclidean distance between two vectors x and y
        x = np.array(x)
        y = np.array(y)
        euc_distance = np.sqrt(np.sum((x - y) ** 2))
        return euc_distance

    def vector_mean(self,vectors):
        #TODO calculate mean of the list of vectors
        #TODO you can use the numpy library to automatically get the mean of some vectors
        vectors = np.array(vectors)
        return list(np.mean(vectors, axis=0))


    def classify(self,input):
        #TODO 1.calculate Euclidean distances between input and the means and
        # 2. return the mean index with min distance
        input = np.array(input)
        distances = [self.euclidian_distance(input, mean) for mean in self.means]
        min_mean = np.argmin(distances)
        return min_mean

    def train(self, inputs):
        # choose k random points as the initial means
        #self.means = random.sample(inputs, self.k)#step 1
        # in order for the unittest to work, we need specific predefined points
        self.means = [inputs[32], inputs[67], inputs[46]]

        assignments = None
        iter = 0
        while iter != 100:
            # find new assignments
            assignments = list(map(self.classify, inputs))

            # compute new means based on the new assignments
            for i in range(self.k):
                # find all the points assigned to cluster i
                i_points = [p for p, a in zip(inputs,assignments) if a == i]
                if i_points:
                    # make sure i_points is not empty so don't divide by 0
                    self.means[i] = self.vector_mean(i_points)
            iter += 1
