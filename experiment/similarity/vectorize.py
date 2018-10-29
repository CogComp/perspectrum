import sent2vec
import sys

def


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python ... [perspectives] [sent2vec_model] [out]", file=sys.stderr)
        exit(1)

    persp_file = sys.argv[1]
    sent2vec_model = sys.argv[2]
    out_path = sys.argv[3]

    model = sent2vec.Sent2vecModel()
    model.load_model(sent2vec_model)

    