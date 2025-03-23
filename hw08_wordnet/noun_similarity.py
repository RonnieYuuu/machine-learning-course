import nltk
from nltk.corpus import wordnet as wn


def get_similarity_scores(pairs):

    results = []

    for pair in pairs:
        word1, word2 = pair

        # get all the synsets of word1 and word2

        synsets_word1 = wn.synsets(word1)
        synsets_word2 = wn.synsets(word2)

        max_score = 0.0
        max_line = () #should look like "('food-fruit', 0.1)"

        #TODO 1. iterate over all combinations of synsets formed by the synsets of the words in the word pair
        #TODO 2. determine the maximum similarity score
        #TODO 3. save max_line in results in form ("word1-word2", similarity_value) e.g.('car-automobile', 1.0)
        #TODO 4. return results in order of decreasing similarity
        for syn1 in synsets_word1:
            for syn2 in synsets_word2:
                # calculate path similarity of synset
                similarity = syn1.path_similarity(syn2)
                if similarity is not None and similarity > max_score:
                    max_score = similarity
                    max_pair = f"{word1}-{word2}"
        max_line += (max_pair, max_score)
        # ("word1-word2", max_similarity)
        results.append(max_line)

    # decreasing
    results.sort(key=lambda x: x[1], reverse=True)

    return results



