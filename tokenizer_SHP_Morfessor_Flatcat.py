# IDEAS
# 1. BPE with better vocab size!!
# 2. Morfessor FlatCat
# 3. Unigram with better PMI

import score
import re
import morfessor

"""
Morfessor steps
a. Initialize an I object, call read_corpus_file()
b. Initialize a BaselineModel()
c. load_data() that we read above
d. Call train_batch()
e. write_binary_file()

(Optional)
f. read_binary_model_file() and evaluate!!


"""

def formatter(subwords):
    a = subwords
    # a = ' '.join(a)

    a = a.replace(' ', " + ")
    a = '1 ' + a
    

    return a

import flatcat

# lang = "shp"

# PHASE 1 - Morfessor Baseline Training


always_split_regex = r'ib|im|ri|ia'
always_split_regex = None
never_split_regex = r'ku|ru|pi|ar|iw|ra|mi\
                        |sa|ik|uk|ch|be|en|\
                        pá|né|ar|si|ni|ki|\
                          hi| ná| ma|ti|má'
never_split_regex = None

def subword_tokenize(lang, train_file, input_file, output_file, print_result = False, write_output = False):

    print("_______________________________________________________________________________")
    print("PHASE 1")
    io = morfessor.MorfessorIO()
    train_data = list(io.read_corpus_file(train_file))
    model_tokens = morfessor.BaselineModel(
                                        nosplit_re=never_split_regex,
                                        forcesplit_list=always_split_regex)
    model_tokens.load_data(train_data, count_modifier = lambda x:1)
    model_tokens.train_batch(algorithm = 'recursive')

    lines = []

    for word in open(input_file):
        pred = " ".join(model_tokens.viterbi_segment(word)[0]).strip()
        lines.append(formatter(pred))

    with open(f'{lang}_seg_output.txt', 'w+') as f:
        f.write("\n".join(lines))

    print("Morfessor Baseline Training Complete")

    # PHASE 2 - Flatcat Training
    
    print("_______________________________________________________________________________")
    print("PHASE 2")


    segment_file = f'{lang}_seg_output.txt'

    io = flatcat.FlatcatIO()
    morph_usage = flatcat.categorizationscheme.MorphUsageProperties()
    model = flatcat.FlatcatModel(morph_usage, 
                                corpusweight = 1.0, 
                                nosplit=never_split_regex,
                                forcesplit=always_split_regex)
    model.add_corpus_data(io.read_segmentation_file(segment_file))
    model.initialize_baseline()
    model.initialize_hmm()

    print("Log: Flatcat HMM Initialized")

    # PHASE 3 - Prediction and Scoring!

    print("_______________________________________________________________________________")
    print("PHASE 3")
    our_output_file = f'pred_{lang}.test_MFC.tgt'

    i = -1

    words = score.read_file(input_file)
    lines = []
    with open(our_output_file, 'w+') as f:
        for word in words:

            # print(word)
            # seg = model.viterbi_segment(word)[0]
            # print(seg)
            line = ' '.join(model.viterbi_segment(word)[0])
            # print(line)
            # print(seg)
            lines.append(line)
            # f.write(line + '\n')
                    
            i+=1

    if(write_output):
        with open(our_output_file, 'w+') as f:
            f.write("\n".join(lines))
        print("Output written!")

    print("Log: Subword Tokenization and Evaluation")

    if(print_result):
        golds = score.read_file(gold_file)

        print(score.evaluate(golds, lines))



if __name__ == "__main__":

    lang = "shp"

    train_file = f'miniproj1-dataset/{lang}.train.src'
    # segment_file = f'miniproj1-dataset/{lang}.train.tgt'
    input_file = f'miniproj1-dataset/{lang}.dev.src'
    gold_file = f'miniproj1-dataset/{lang}.dev.tgt'

    subword_tokenize(lang, train_file, input_file, gold_file, print_result=True, write_output=False)
    print()
    print()
    print()
    test_file = f'miniproj1-dataset/{lang}.test.src'
    subword_tokenize(lang, train_file, test_file, gold_file, print_result=False, write_output=True)

    print("MORFESSOR DONE!!")