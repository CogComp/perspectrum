import pickle as pkl
import math
import collections

from enum import Enum


class _Mode(Enum):
    PREP = 0
    READY = 1


class SparseVector:
    """
    A simple dictionary-based sparse vector implementation
    """
    def __init__(self):
        self._vec = {}

    def dot(self, vec):
        _result = 0
        for key, val in self._vec.items():
            if key in vec:
                _result += vec[key] * val
        return _result

    def __getitem__(self, index):
        if index in self._vec:
            return self._vec[index]
        else:
            return 0

    def __setitem__(self, index, value):
        if value != 0:
            self._vec[index] = value


class IdfProcessor:
    """
    Process tfidf counts

    Definition of idf used here: log10(N/k). where N = total # of docs, k = # of docs the target word appears in
    """
    def __init__(self):

        self.num_doc = 0
        self.idf = collections.OrderedDict()
        self._mode = _Mode.PREP

    def add_doc(self, tok_list):
        """
        Add a document to collection for idf computation.
        Note this function doesn't actually invoke idf computation. Call compute_idf() when all docs have been added
        :param tok_list: list of tokens in this document
        :return:
        """
        if self._mode != _Mode.PREP:
            raise ValueError("Curent Tfidf object is in immutable state. Can't add new doc.")

        self.num_doc += 1
        tok_set = set(tok_list)
        for tok in tok_set:
            self.idf[tok] = self.idf.get(tok, 0) + 1

    def compute_idf(self):
        """

        :return:
        """
        for key in self.idf:
            self.idf[key] = math.log10(self.num_doc / self.idf[key])

        self._mode = _Mode.READY

    def save(self, path):
        """
        Save idf into a pickle file
        :return:
        """
        with open(path, 'wb') as fout:
            pkl.dump(self.idf, fout)


class Tfidf:

    def __init__(self, pkl_path, oov_idf=None):
        """
        Loads pre-processed idf (See IdfProcessor class above) from pkl file
        :param pkl_path: idf pickle
        :param oov_idf: Default idf weight for unseen words. If not set to a specific value, it will be set to the
        maximum idf weight present in the input pickle.
        """
        with open(pkl_path, 'rb') as fin:
            self.idf = pkl.load(fin)
            if oov_idf is None:
                self.oov_idf = max(self.idf.items(), key=(lambda item: item[1]))[1]
            elif oov_idf < 0:
                raise ValueError("oov_idf must be non-negative.")
            else:
                self.oov_idf = oov_idf

    def get_word_tfidf(self, word, tf):
        """
        Get tfidf weight of a word, given its surface form and term frequency
        :param word:
        :param tf: (normalized) term frequency of the word
        :return:
        """
        return tf * self._get_idf(word)

    # TODO: Think of a better name for the function. (wish I was writing java :p)
    def get_word_tfidf_in_doc(self, word, all_toks_in_doc):
        """
        Get tfidf weight of a word, given list of all tokens in document
        :param word:
        :param all_toks_in_doc:
        :return:
        """
        return self._compute_tf(word, all_toks_in_doc) * self._get_idf(word)

    def _get_idf(self, word):
        if word in self.idf:
            return self.idf[word]
        else:
            return self.oov_idf

    def _compute_tf(self, tok, all_toks):
        """
        helper function to compute term frequency of token normalized by document length
        :param tok: a token
        :param all_toks: The document (represented by a list of tokens)
        :return: term frequency of tok, normalized by document length
        """
        return all_toks.count(tok) / len(all_toks)


if __name__ == '__main__':

    pkl_p = "/shared/preprocessed/schen149/newsdata/nyt_tfidf/nyt_tfidf.pkl"
    tfidf = Tfidf(pkl_p)

    # Example 1: Stop word
    doc1 = ["This", "is", "a", "sample", "a"]
    word1 = "This"
    print("Tfidf for \"{}\" in doc1:\t {}".format(word1 ,tfidf.get_word_tfidf_in_doc(word1, doc1)))

    # Example 2: NE
    doc2 = ["This", "is", "Donald", ",", "Mr.", "Donald", "."]
    word2 = "Donald"
    print("Tfidf for \"{}\" in doc2:\t {}".format(word2 ,tfidf.get_word_tfidf_in_doc(word2, doc2)))

    # Example 3: Word that's not present in current document
    word3 = "Gibberish"
    print("Tfidf for \"{}\" in doc2:\t {}".format(word3 ,tfidf.get_word_tfidf_in_doc(word3, doc2)))

    # Example 4: Unseen word during idf calculation
    doc3 = ["This", "is", "a", "AClearlyMade-upWord", ",", "Mr.", "Donald", "."]
    oov_word = "AClearlyMade-upWord"
    print("Tfidf for \"{}\" in doc3:\t {}".format(oov_word, tfidf.get_word_tfidf_in_doc(oov_word, doc3)))

