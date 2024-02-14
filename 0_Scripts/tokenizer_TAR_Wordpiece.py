# IDEAS
# 1. BPE with better vocab size!!
# 2. Morfessor FlatCat
# 3. Unigram with better PMI

import score
import re
import morfessor
from transformers import AutoTokenizer
import torch

from Timer import Timer

# HELPER FUNCTIONS
def clean_segment(model_name, text):
    if model_name == "xlm-roberta-base":
        return remove_underscore(text)
    elif model_name == "bert-base-cased":
        return remove_hash(text)

def remove_hash(text):
    if(text[0] == "#"):
        return text[2:]
    return text

def remove_underscore(text):
    # print("called")

    if(text[0] == "▁"):
        # print("inside")
        # print(text[1:])
        return text[1:]
    return text

def print_status(before, after, language, model_name, vocab_size, batch_size, timer):
    h_dic = {"Language": language, "Model": model_name, "V": vocab_size, "B":batch_size}
    print(h_dic)

    if(before):
        print("BEFORE: ", before)
    if(after):
        print("AFTER: ", after)
    print(timer)
    print("___________________________________________")


# HYPERPARAMETER OPTIMIZATION ON DEV SET
def hyperparameter_optimizer(hyperparam_dictionary, lang = "tar"):

    print("Beginning Hyperparameter Optimization")
    print("_________________________________________________________________")
    language = lang
    for model_name in hyperparam_dictionary["model_name"]:
        for vocab_size in hyperparam_dictionary["vocab_size"]:
            for batch_size in hyperparam_dictionary["batch_size"]:
                before, after, timer = subword_tokenizer(model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size)
                
                print_status(before, after, model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size, timer=timer)
                
    print("_________________________________________________________________")
    print("All Hyperparameters Tested!")

    return 0

# TRAINING ON TRAIN SET
def subword_tokenizer(model_name, language, vocab_size, batch_size, dev_version = True):

    timer = Timer()
    timer.start()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(type(tokenizer))

    lang = language

    train_file = f'miniproj1-dataset/{lang}.train.src'
    segment_file = f'miniproj1-dataset/{lang}.train.tgt'
    if(dev_version):
        input_file = f'miniproj1-dataset/{lang}.dev.src'
    else:
        input_file = f'miniproj1-dataset/{lang}.test.src'
    gold_file = f'miniproj1-dataset/{lang}.dev.tgt'


    def get_training_corpus():
        # dataset = score.read_file(segment_file) + score.read_file(gold_file)
        dataset = score.read_file(segment_file)
        for start_idx in range(0, len(dataset), batch_size):
            samples = dataset[start_idx : start_idx + batch_size]
            yield samples

    train_corpus = get_training_corpus()
    test_words = score.read_file(input_file)

    new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=train_corpus, vocab_size=vocab_size)

    preds = []
    for w in test_words:
        seg = new_tokenizer.tokenize(w)
        seg = ' '.join([clean_segment(model_name, s) for s in seg])
        preds.append(seg)

    if(dev_version):
        old_preds = []
        for w in test_words:
            # print(w)
            seg = tokenizer.tokenize(w)
            seg = ' '.join([clean_segment(model_name, s) for s in seg])
            # print(seg)
            old_preds.append(seg)

        # print()
    
    
        golds = score.read_file(gold_file)
        # print("WITHOUT TRAINING:-")
        before = score.evaluate(golds, old_preds)
        # print(before)



        # print("WITH TRAINING:-")
        after = score.evaluate(golds, preds)
        # print(after)

    timer.stop()


    if(dev_version == False):
        with open(f'pred_{lang}.test_BBC.tgt', 'w+') as f:
            f.write("\n".join(preds))

    if(dev_version):
        return before, after, str(timer)
    else:
        return None, None, str(timer)


hyperparams = {
    "language": ["tar", "shp"],
    "model_name": ["bert-base-cased", "xlm-roberta"],
    "vocab_size": list(range(300, 1300, 100)),
    "batch_size": [4, 16, 32, 64, 128]

}


if __name__ == "__main__":
    # hyperparameter_optimizer(hyperparams)

    # TAR Test file!!
    model_name = 'bert-base-cased'
    language = 'tar'
    vocab_size = 900
    batch_size = 128
    dev_version = False
    before, after, timer = subword_tokenizer(model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size,
                       dev_version=True)
    print_status(before, after, model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size, timer=timer)
                
    
    before, after, timer = subword_tokenizer(model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size,
                       dev_version=False)
    print_status(before, after, model_name = model_name,
                       language = language, 
                       vocab_size = vocab_size,
                       batch_size = batch_size, timer=timer)
                

    # SHP Test file!!

    # model_name = 'bert-base-cased'
    # language = 'shp'
    # vocab_size = 1200
    # batch_size = 128
    # before, after, timer = subword_tokenizer(model_name = model_name,
    #                    language = language, 
    #                    vocab_size = vocab_size,
    #                    batch_size = batch_size,
    #                    dev_version=dev_version)
    # print_status(before, after, model_name = model_name,
    #                    language = language, 
    #                    vocab_size = vocab_size,
    #                    batch_size = batch_size, timer=timer)

 
    print("WORDPIECE DONE!!")

# Best Hyperparams
    # or V = 32
# {'Language': 'tar', 'Model': 'bert-base-cased', 'V': 900, 'B': 128}
# BEFORE:  {'f1': 0.19, 'precision': 0.143, 'recall': 0.286}
# AFTER:  {'f1': 0.689, 'precision': 0.626, 'recall': 0.766}
# 0.115 secs
    
    

# SENTENCE_PIECE:  0.084 secs
# MORFESSOR:  2.859 secs
