# This exercise contains tasks to practice comprehensions.
# With these, multiple-line for-loop constructions can be expressed in expressive one-liners.
# In the following tasks, you are not allowed to use: list(), dict() and set()


class list:
    def __init__(self):
        raise NotImplementedError("Lists constructor is not allowed in this exercise, use list comprehensions.")

class dict:
    def __init__(self):
        raise NotImplementedError("Dict constructor is not allowed in this exercise, use dict comprehensions.")
    
class set:
    def __init__(self):
        raise NotImplementedError("Set constructor is not allowed in this exercise, use sets comprehensions.")


def multiply_by(x, list1):
    """
    Multiplies each value in list1 by x and returns it as a new list.
    """
    new_list = []
    for i in list1:
        new_value = i * x
        new_list.append(new_value)
    return new_list

def get_longest_word(list):
    """
    Given a list of sentences, return a list of the longest word (tokenized just with white space) of each sentence.
    """
    longest_words = []
    for sent in list:
        tokens = sent.split(" ")
        max_len = 0
        for token in tokens:
            if len(token) > max_len:
                max_len = len(token)
                longest_word = token
        longest_words.append(longest_word)
    return longest_words

def merge_lists(nested_lists):
    """
    Given a list of lists, returns a list containing all the elements of the sublists, flattened into a single list.
    """
    result = []
    for sublist in nested_lists:
        for elem in sublist:
            result.append(elem)
    return result

def transpose_matrix(matrix):
    """
    given a 2D matrix (a list of lists where each inner list represents a row of the matrix),
    returns the transposed matrix.
    """
    i = 0
    expected = []
    while i < len(matrix):
        sub_expected = []
        for sublist in matrix:
            sub_expected.append(sublist[i])
        i += 1
        expected.append(sub_expected)
    return expected

def map_zip(list1, list2):
    """
    It should return a dictionary mapping the 'nth' element in list1 to the 'nth' element in list2.
    Make use of the 'zip()' function in your dictionary comprehension, that can handle lists of different sizes
    automatically.
    """
    result = {}
    x = zip(list1, list2)
    for t in x:
        result[t[0]] = t[1]
    return result

def count_occur(dict1):
    """
    Given a dictionary of words as keys and its frequencies as values, return the total counts of words with prefix "anti".
    """
    summe = 0
    for word in dict1:
        if word.startswith("anti"):
            summe += dict1[word]
    return summe

