
import sentencepiece as spm
import score

def optimize():
    lang = "shp" 
    model_type = "unigram" # Best
    # model_type = 'char' # Horrible
    # model_type = 'bpe' # Ok
    vocab_size=300
    if lang == "tar":
        vocab_sizes = [100, 200, 300, 400, 500, 593]
        vocab_sizes = list(range(100, 600, 50))
    else:
        vocab_sizes = [100, 200, 300, 400, 500, 600, 700]
        vocab_sizes = list(range(100, 750, 50))

    train_file = f'miniproj1-dataset/{lang}.train.src'


    input_file = f'miniproj1-dataset/{lang}.dev.src'
    output_file = f'miniproj1-dataset/{lang}.dev.tgt'

    results = []
    for vocab_size in vocab_sizes:


        spm.SentencePieceTrainer.Train(
            input=train_file,
            model_prefix=f'unigram_{lang}',
            vocab_size=vocab_size,
            model_type=model_type
        )
        s = spm.SentencePieceProcessor(model_file=f'unigram_{lang}.model')

        golds = []
        preds = []

        for word, morph in zip(open(input_file), open(output_file)):
            gold = morph.strip()
            pred = ' '.join(s.encode(word, out_type=str))[1:].lstrip()
            golds.append(gold)
            preds.append(pred)

        
        results.append(score.evaluate(golds, preds))
        # print(, "log.txt")

    print()
    print("LANGUAGE: ", lang)
    print("MODEL TYPE: ", model_type)
    for vocab_size, result in zip(vocab_sizes, results):
        print("VOCABULARY SIZE: " + str(vocab_size))
        print(result)
        

# with open(f'pred_{lang}.dev.tgt', 'w') as f:
#   f.write("\n".join(preds))
        

if __name__ == "__main__":
    optimize()