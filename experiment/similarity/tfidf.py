import pickle as pkl
import math
import collections
from enum import Enum

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize


class _Mode(Enum):
    PREP = 0
    READY = 1


class SparseUnigramDoc:
    """
    A simple sparse unigram counts representation of document
    """
    def __init__(self, data=None):
        if data:
            self._dict = data
        else:
            self._dict = {}

    @property
    def dict(self):
        return self._dict

    def dot(self, doc2):
        _result = 0
        for key, val in self._dict.items():
            if key in doc2.dict:
                _result += doc2.get(key) * val
        return _result

    def get(self, word):
        if word in self._dict:
            return self._dict[word]
        else:
            return 0

    def set(self, word, score):
        if word != 0:
            self._dict[word] = score

    def magnitude(self):
        _result = 0
        for key, val in self._dict.items():
            _result += val * val
        return math.sqrt(_result)

    def cos_similiarity(self, doc):
        return self.dot(doc) / (self.magnitude() * doc.magnitude())


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
        self.lem = WordNetLemmatizer()

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

    def vectorize(self, text):
        """
        Yeah just vectorize it
        :param text: piece of untokenized text
        :return: A SparseUnigram instance, storing the tfidf representation of the input text(see above)
        """
        toks = [self.lem.lemmatize(t) for t in word_tokenize(text)]
        doc_len = len(toks)
        doc = SparseUnigramDoc()
        counter = collections.Counter(toks)
        for key, val in counter.items():
            doc.set(key, val / doc_len * self._get_idf(key))

        return doc


if __name__ == '__main__':

    pkl_p = "/shared/preprocessed/schen149/newsdata/nyt_tfidf/nyt_tfidf.pkl"
    tfidf = Tfidf(pkl_p)

    text1 = "My name is Rick."
    text1_a = "My name is also Rick."
    text2 = "Das ist alles Gibbbbbeeerish."

    t1 = tfidf.vectorize(text1)
    t1_a = tfidf.vectorize(text1_a)
    t2 = tfidf.vectorize(text2)

    print(t1.cos_similiarity(t1_a))
    print(t2.cos_similiarity(t1))

