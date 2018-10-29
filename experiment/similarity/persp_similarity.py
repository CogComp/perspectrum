from .tfidf import Tfidf
import sys

class PerspSim:
    """
    TF-IDF baseline for finding relevant perspective/evidence given claim/perspective
    """

    def __init__(self, evidence_pool, preprecoessed_idf):
        """

        """
        self._claim = claim
        self._evidence_pool = evidence_pool
        self.tfidf = Tfidf(preprecoessed_idf)


    def

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python ... [perspectives] [sent2vec_model] [out]", file=sys.stderr)
        exit(1)

    persp_file = sys.argv[1]
    sent2vec_model = sys.argv[2]
    out_path = sys.argv[3]




