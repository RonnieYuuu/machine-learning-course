import nltk.corpus
from nltk.tokenize import * #这样就已经import sent_tokenize了
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import bigrams
#import nltk.data
from nltk.corpus import stopwords, inaugural, words
#from nltk.corpus import webtext
from nltk import FreqDist
from nltk.corpus import swadesh


#import string


class Preprocessor(object):
    def __init__(self, text, text_id):
        """reads the file text as a string"""
        self.text = text
        self.text_id = text_id
        self.stopwords = set(stopwords.words("english"))
        self.english_vocab = set(words.words())
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    @classmethod
    def from_nltk_book(cls):
        """Reads the inaugural text from the nltk book as a string. Note that this is a @classmethod
        which will be used instead of the default constructor when creating the object. Remember to
         return cls(something)"""
        inaugural_text = '\n'.join(inaugural.raw(fileid) for fileid in inaugural.fileids())
        #inaugural_text = " ".join(inaugural.words())
        return cls(inaugural_text, "inaugural")
    #未更正

    @classmethod
    def from_text_file(cls, path, text_id):
        """Reads the text from given file as a string. Note that this is a @classmethod
        which will be used instead of the default constructor when creating the object.
         Remember to return cls(something)"""
        with open(path, "r") as file:
            file_text = file.read()
        return cls(file_text, text_id)
    #已更正

    def get_no_of_sents(self):
        """Split the input text into sentences and return the length of the text in sentences."""
        tokens = sent_tokenize(self.text)
        return len(tokens)
    #避免在function里面import资源，如from nltk.tokenize import sent_tokenize

    def tokenize_text(self):
        """Split the text into tokens. Use an nltk method for that, equivalent to s.split()"""
        tokens = word_tokenize(self.text)
        return tokens

    def stem_token(self, token):
        """Stem the given token with Porter Stemmer after converting it to lowercase."""
        stemmer = PorterStemmer()
        return stemmer.stem(token.lower())

    def lemmatize_token(self, token):
        """Lemmatize the given token, after converting it to lowercase."""
        return self.lemmatizer.lemmatize(token.lower())

    def get_20_most_frequent_words(self):
        """
        Return the 20 most frequent words of the text after removing the stopwords and all non-alphanumeric characters
         (in other words, remove all punctuation). Note that the stopwords are stored with their lemmata so you
         will have to make sure you check whether the lemma of each word is contained in the stopwords list.
         Make sure you use the method tokenizeText() that you implemented above
         """
        tokens = [self.lemmatize_token(t.lower()) for t in self.tokenize_text() if t.isalnum()]
        filtered_tokens = [t for t in tokens if t not in self.stopwords]
        freq_dist = FreqDist(filtered_tokens)
        print(freq_dist.most_common(20))
        return freq_dist.most_common(20)


    def get_bigram_words(self, token_a):
        """
        Return the 3 most frequent words of the text that follow the word token_a. Preprocessing: Lowercase all words,
        remove all stopwords and all non-alphanumeric characters. Do not apply lemmatization.
        Make sure you use the method tokenize_text() that you implemented above.
        """
        stopwords = set(nltk.corpus.stopwords.word("english"))
        lowercase = [word.lower() for word in self.tokenize_text()]
        filtered_text = [word for word in lowercase if word.isalnum() and word not in stopwords]
        bigram_tokens = list(bigrams(filtered_text))
        cfd = nltk.ConditionalFreqDist(bigram_tokens)
        max_three = cfd[token_a].most_commen(3)
        return set(k[0] for k in max_three)
        #已更正

        #tokens = [t.lower() for t in self.tokenize_text() if t.isalnum()]
        #filtered_tokens = [t for t in tokens if t not in self.stopwords]
        #bigrams_list = list(bigrams(filtered_tokens))
        #following_words = [bigram[1] for bigram in bigrams_list if bigram[0] == token_a]
        #freq_dist = FreqDist(following_words)
        #return freq_dist.most_common(3)

    def get_least_frequent_unusual_words(self):
        """
        Return the set of the least frequent unusual words in the text. A token is considered a word if it contains no
        non-alphabetic characters. A word is considered unusual if it has more than 14 characters and does not appear
        in the "English vocabulary".
        """
        tokens = [t.lower() for t in self.tokenize_text() if t.isalpha()]
        filtered_tokens = [t for t in tokens if t not in self.english_vocab and len(t) > 14]
        freq_dist = FreqDist(filtered_tokens)
        min_count = min(freq_dist.values())
        least_frequent_words = [word for word, count in freq_dist.items() if count == min_count]
        return least_frequent_words

    def get_originality_score(self, most_freq_words):
        """
        Return the originality score of a text. This score can be measured by counting how many words contained
        in the swadesh list of nltk are also contained in the list with the 20 most frequent words of the text.
        For example, if 10 words can be found in both lists, the originality score is 10.
        The list with the 20 most frequent words should be given to the method as a parameter.
        """
        swadesh_list = set(swadesh.words('en'))
        tokens = [t.lower() for t in self.tokenize_text() if t.isalnum()]
        filtered_tokens = [t for t in tokens if t not in self.stopwords]
        freq_dist = FreqDist(filtered_tokens)
        return len([word for word, _ in most_freq_words if word in swadesh_list])

    def get_no_of_keywords_sents(self, keywords_list):
        """
        return a dictionary where each key is a keyword,
        and the corresponding value is the number of sentences in the text that contain that keyword.
        The list of keywords should be provided as an argument to the method.
        Note that you don’t need to deal with case.
        """
        sentences = sent_tokenize(self.text)
        keyword_sent_counts = {keyword: 0 for keyword in keywords_list}
        for keyword in keywords_list:
            for sentence in sentences:
                if keyword in sentence:
                    keyword_sent_counts[keyword] += 1
        return keyword_sent_counts
        #已更正

if __name__ == '__main__':
    """Create two instances of the Preprocessor class. One instance should be created through the text file 
    ada_lovelace.txt and the other one through the inaugural text included in the nltk book. """

    """For each instance of the Preprocessor, do the following:
      a) print out the numbers of sentences of the text
      b) print out the 20 most frequent words of the text
      c) print out the originality score of the text."""


    ada_instance = Preprocessor.from_text_file('ada_lovelace.txt', 'ada_lovelace')


    inaugural_instance = Preprocessor.from_nltk_book()
