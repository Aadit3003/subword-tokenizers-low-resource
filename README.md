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
I used the Wordpiece tokenization algorithm, particularly the BertTokenizerFast() implementation from Huggingface ('bert-base-cased') for this task. I also tried using the 'xlm-roberta' tokenizer based on Sentencepiece, but it consistently underperformed as compared to the BertTokenizer. I performed grid search hyperparameter tuning for the vocabulary size and batch size, finally deciding on a vocabulary size of 900, and a batch size of 128. The runtime was roughly 0.12 seconds, slightly longer than SentencePiece, but much shorter than Morfessor. For tar it outperformed both baselines by a relatively large margin. However, this algorithm didn't seem to work as well for shp.

### Morfessor Flatcat for Shipibo-Konibo (shp) ([code](https://github.com/Aadit3003/subword-miniproject-1/blob/7dd2aca9aecd7bf9b28f99a1d81c56613a8b4cd4/0_Scripts/tokenizer_SHP_Morfessor_Flatcat.py))
Building on the baseline code provided for Morfessor, I trained a baseline Morfessor model (recursive algorithm) on the train set and performed segmentation using the baseline model. I added this corpus of the baseline segmented data to a Flatcat model and initialized its Hidden Markov Model. This was followed by using the viterbi segment method and evaluation. The improvements over the baseline Morfessor were not too significant, but it beat the baselines.

I experimented with using the "force_split" and "no_split" parameters by observing morphemes in the train and dev target files and populating these lists. I included some common character pairs that appeared at morpheme boundaries to the force_split list (for both the Morfessor baseline and Flatcat models). Similarly, I added character pairs that always occurred internal to the morphemes to the no_split list. This seemed to improve performance slightly, however, adding too many character pairs severely degraded the performance.
