import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
# !pip install transformers==4.5.0 fugashi==1.1.0 ipadic==1.0.0
class FillHolefromBertMaskedLM:
  def __init__(self, model_name="cl-tohoku/bert-base-japanese-whole-word-masking", cuda=False):
    """
    BertMaskedLMを使って文章の穴埋め問題を事前学習により推論します。
    モデル内では
      (文章数, 単語数, 単語のトークン数ここでは1) -> (文章数, 単語数, 32000次元ベクトル)
    を行うことで各トークンをbertに登録されている32000語への確率として出力します。
    
    """
    self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    self.cuda = cuda
    print(self.bert_mlm.config)
  def _random_masked(self, text: str, num_topk: int) -> str:
    encoding = self.tokenizer.tokenize(text)
    i = np.random.randint(0, len(encoding), 2).tolist()
    for idx in range(len(i)):
      encoding[i[idx]] = "[MASK]"
    dummy_text = "".join(encoding)
    print_text = dummy_text.replace("[MASK]", "○")
    print(f"({print_text})の文章から○の部分を{num_topk}パターン予測します")
    print("~"*100)
    return dummy_text
  def _predict_mask_topk(self, text, num_topk):
    """
    文章中の最初の[MASK]をスコアの上位のトークンに置き換える。
    上位何位まで使うかは、num_topkで指定。
    出力は穴埋めされた文章のリストと、置き換えられたトークンのスコアのリスト。
    """
    # 文章を符号化し、BERTで分類スコアを得る。
    # random_masked_text = self._random_masked(text, num_topk)
    input_ids = self.tokenizer.encode(text, return_tensors='pt')
    if self.cuda:
        input_ids = input_ids.cuda()
        self.bert_mlm = self.bert_mlm.cuda()
    with torch.no_grad():
        output = self.bert_mlm(input_ids=input_ids)
    scores = output.logits # (文章数, 単語数, 32000)
    # スコアが上位のトークンとスコアを求める。
    mask_position = input_ids[0].tolist().index(4)  # [MASK]の最初のindex数のみ取得する
    topk = scores[0, mask_position].topk(num_topk) # 文章中の[MASK]のベクトルからスコア順に取得

    ids_topk = topk.indices # トークンのID
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk) # トークン ["予測単語", "", ""]
    scores_topk = topk.values.cpu().numpy() # スコア

    # 文章中の[MASK]を上で求めたトークンで置き換える。
    text_topk = [] # 穴埋めされたテキストを追加する。
    for token in tokens_topk:
        token = token.replace('##', '')
        text_topk.append(text.replace('[MASK]', token, 1))
    for text, score in zip(text_topk, scores_topk):
        print(f"prediction sequence: {text} -- score: {score}", sep="\n")
    return text_topk, scores_topk

  def greedy_prediction(self, text: str, num_topk: int):
    """
    [MASK]を含む文章を入力として、貪欲法で穴埋めを行った文章を出力する。
    """
    text_copy = self._random_masked(text, num_topk)
    # 前から順に[MASK]を一つづつ、スコアの最も高いトークンに置き換える。
    for _ in range(text_copy.count('[MASK]')):
        print(f"{_+1}番目:")
        text_copy = self._predict_mask_topk(text_copy,
                                      num_topk)[0][0]
    print(f"最終: {text_copy}")

  def beam_search(self, text: str, num_topk: int):
    """
    ビームサーチで文章の穴埋めを行う。
    """
    text_copy = self._random_masked(text, num_topk)
    num_mask = text_copy.count('[MASK]')
    text_topk = [text_copy]
    scores_topk = np.array([0])
    for _ in range(num_mask):
        print(f"{_+1}番目:")
        # 現在得られている、それぞれの文章に対して、
        # 最初の[MASK]をスコアが上位のトークンで穴埋めする。
        text_candidates = [] # それぞれの文章を穴埋めした結果を追加する。
        score_candidates = [] # 穴埋めに使ったトークンのスコアを追加する。
        for text_mask, score in zip(text_topk, scores_topk): # num_topkの分だけループする
            if _ > 0:
              print(f"{text_mask}をもとに予測を開始します....")
            text_topk_inner, scores_topk_inner = self._predict_mask_topk(
                text_mask, num_topk
            )
            text_candidates.extend(text_topk_inner)
            score_candidates.append( score + scores_topk_inner )

        # 穴埋めにより生成された文章の中から合計スコアの高いものを選ぶ。
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk] # scoreから合計が高いindexの上位１０を取得する
        # 次のループに回す
        text_topk = [ text_candidates[idx] for idx in idx_list ] # scoreの高かった１０の穴埋め済み文章を取得して次の入力とする
        scores_topk = [ score_candidates[idx] for idx in idx_list] # 前回までのスコアの累積の上位を取得する

    print(f"最終: {text_topk[0]}")