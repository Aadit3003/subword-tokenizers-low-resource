# subword-miniproject-1
Mini-Project 1 for 11824. I use two subword tokenizers using Morfessor Flatcat and Wordpiece for Shipibo-Konibo (shp) and Rarámuri (tar) respectively (to beat Morfessor and Sentencepiece baselines).
 ## File Structure
The files are organized as follows:
- 0_Scripts:- Contains the two tokenizer files for shp and tar respectively. Also contains utilities like evaluation (score.py) and timer (Timer.py)
- 1_Final_Submission_Files:- The predictions of these tokenizers on the test files for shp and tar.
- 2_Log_Files:- The output logs from training the two tokenizers.
- 3_Previous_Attempts:- A previous attempt using SentencePiece that failed to beat baseline performance.
- 4_Given_Baselines:- The notebook provided to calculate scores.

 ## Run Commands
 Any one of these commands within the 0_Scripts directories will produce the output and log files for both tokenizers
`sbatch run.sh` \
or \
`python tokenizer_SHP_Morfessor_Flatcat.py` and
`python tokenizer_TAR_Wordpiece.py`

 ## My Approach
### Wordpiece Tokenization for Rarámuri (tar) ([code](https://github.com/Aadit3003/subword-miniproject-1/blob/7dd2aca9aecd7bf9b28f99a1d81c56613a8b4cd4/0_Scripts/tokenizer_TAR_Wordpiece.py))
I used the Wordpiece tokenization algorithm, using the BertTokenizerFast() implementation from Hugginface ('bert-base-cased') for this task. I also tried using the 'xlm-roberta' tokenizer based on Sentencepiece, but it consistently underperformed the BERT tokenizer. I did a grid search for hyperparameter optimization and landed on Vocabulary size of 900, and batch size of 128. The runtime was a bit longer than the SentencePiece baseline, but much smaller than Morfessor, and it outperforms both by a relatively large margin. Wordpiece however failed to surpass baselines for shp.


### Morfessor Flatcat for Shipibo-Konibo (shp) ([code](https://github.com/Aadit3003/subword-miniproject-1/blob/7dd2aca9aecd7bf9b28f99a1d81c56613a8b4cd4/0_Scripts/tokenizer_SHP_Morfessor_Flatcat.py))
Building on the baseline code provided for Morfessor, I trained a baseline Morfessor model (recursive algorithm) on the train set. This was followed by adding the corpus data to a Flatcat model and initializing its Hidden Markov Model. This was followed by using the viterbi segment method and evaluation. The improvements over the baseline Morfessor were not too significant, but it beat the baselines. This method didn't seem to work too well for tar.
