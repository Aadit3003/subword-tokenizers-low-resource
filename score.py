import argparse
import sentencepiece as spm
import morfessor

# Evaluation Function
def evaluate(gold, pred):

  tp = 0
  fp = 0
  fn = 0

  for g, p in zip(gold, pred):
    g_bag = g.strip().split(" ")
    p_bag = p.strip().split(" ")

    tp += sum([1 for i in p_bag if i in g_bag])
    fp += sum([1 for i in p_bag if not i in g_bag])
    fn += sum([1 for i in g_bag if not i in p_bag])

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  if precision == 0 or recall == 0:
    f1 = 0
  else:
    f1 = 2 / ((1/precision) + (1/recall))

  return {
      "f1": round(f1, 3),
      "precision": round(precision, 3),
      "recall": round(recall, 3),
  }

def read_file(filename):
    f = open(filename, "r")


    lines = f.readlines()
    # for line in lines:
    #     print(line.strip())

    
    new_actual_words = [aw[:-1] for aw in lines[:-1]]
    new_actual_words.append(lines[-1])
    lines = new_actual_words

    return lines



def print_output(title, language, golds, preds):
    print("__________________________________________________________")
    print(title)
    print("LANGUAGE: ", language)
    # print("GOLDS: ")
    # print(type(golds))
    # print(golds)
    # print("PREDS: ")
    # print(preds)

    print(evaluate(golds, preds))
    print()




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("language", type=str, help='bart or flant5')
  parser.add_argument("gold_file", type=str, help='max length of blog to be generated')
  parser.add_argument("s_pred_file", type=str, help='max length of summary to be generated')
  parser.add_argument("m_pred_file", nargs='?', const = "No file", type=str, help='number of days to generate the blog for')
    # parser.add_argument("past_look_over", type=int, help='number of past summaries to look at!!')
  args = parser.parse_args()

  language = args.language
  gold_file = args.gold_file
  s_pred_file = args.s_pred_file
  m_pred_file = args.m_pred_file  
  print(m_pred_file)

  s_preds = read_file(s_pred_file)
  golds = read_file(gold_file)






  print_output("SENTENCE_PIECE", language, golds, s_preds)
  
  if(m_pred_file is not None):
    m_preds = read_file(m_pred_file)
    print_output("MORFESSOR", language, golds, m_preds)
  
  print("DONE!")
  




