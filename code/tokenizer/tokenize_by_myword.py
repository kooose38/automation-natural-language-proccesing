# !pip install -q janome 
# from janome.tokenize import Tokenzier
import re 
import torch 
from torch.utils.data import DataLoader 
from janome.tokenizer import Tokenizer
import numpy as np 
from create_loader import MakeDataLoader
from preprocessing import PreProcessingTEXT
from typing import Dict, List

class WordToTensorLoader:
    
  def __init__(self):
    self.vocab_size = 0 
    self.word2index = {}
    self.max_len = 0


  def _preprocessing_and_wakati(self, d: dict) -> list:
    text = d["text"]
    text = self._preprocessing(text)
    text = self._make_wakati(text)
    return text 

  def _training_vocab(self, data: list):
      # トークンの辞書作成 ボキャブラリー数 最大系列数
      max_len = 0 # padding 
      vocab_= {} # vocab_size
      word2index = {"<pad>": 0, "<unk>": 1, "<cls>": 2} 
      # pad ... トークンの不足分を補う
      # unk ... 登録されていない未知の単語
      # cls ... 文の末端。文脈ベクトルになりうる位置を示す
      for d in data:
        text = self._preprocessing_and_wakati(d)
        if max_len < len(text):
          max_len = len(text)
        for t in text:
          if t not in  vocab_:
            vocab_[t] = 1
            word2index[t] = len(word2index) 
          else:
            vocab_[t] += 1 
      self.vocab_size = len(vocab_) + 3
      self.word2index = word2index
      self.max_len = max_len 

  def _create_inputs_ids(self, data: list, limit_length: int) -> list:
    # DNN 入力データ作成
    inputs = []
    for d in data:
      sentence2tensor = {}
      label = d["label"]
      text = self._preprocessing_and_wakati(d)
      dummy = []
      label2tensor = torch.tensor([label], dtype=torch.long)
      for r in text:
        if r in self.word2index:
          idx = self.word2index[r]
        else:
          idx = self.word2index["<unk>"]
        dummy.append(idx)
      # max_len から文章の最大トークンの分だけpaddingで埋める
      # paddingを反転させる場合はコメントアウトを入れ替える
      if self.max_len >= len(dummy):
        for _ in range(self.max_len - len(dummy)):
          #####################################################################
          # dummy.insert(0, 0) # この実装は文章の先頭からpaddingで埋めている
          dummy.append(0) # この実装では文章の末尾にpaddingで埋める
          #####################################################################
      else:
        #######################################################################
        # max_len = (-1) * self.max_len
        # dummy = dummy[max_len:]
        dummy = dummy[:self.max_len]
        #######################################################################
      #########################################################################
      # dummy.append(self.word2index["<cls>"]) # 末端に<cls>の追加
      dummy.insert(0, self.word2index["<cls>"]) # 先頭に<cls>の追加
      #########################################################################
      if limit_length != 0:
        #######################################################################
        # limit_length = (-1) * limit_length
        # dummy = dummy[limit_length:]
        #######################################################################
        dummy = dummy[:limit_length]
      sentence2tensor["input_ids"] = torch.tensor(dummy, dtype=torch.long)
      sentence2tensor["labels"] = label2tensor.item()
      inputs.append(sentence2tensor)
    return inputs 

  def _assert_test(self, data: list, inputs: list, limit_length: int):
    # テストコード
    for _ in range(10):
      a, b = np.random.randint(0, len(data), 2).tolist()
      assert inputs[a]["input_ids"].size()[0] == inputs[b]["input_ids"].size()[0]
    if limit_length == 0:
      assert inputs[a]["input_ids"].size()[0] == self.max_len
    else:
      assert inputs[a]["input_ids"].size()[0] == limit_length

  def transform(self, data: List[ Dict[str, str, str, int]], loader: bool=True, batch_size: int=32, train: bool=True, limit_length: int=0):
    """
    data = [
      {"text": "hello world 1", "label": 1},
      {"text": "hello world 2", "label": 0},
      ...
    ]

    loader=Trueでtrain, val, test のLoaderを作成。戻り値は３つ
    loader=Falseでdata単独でinput_idsの作成。戻り値は１つ

    train=Trueでボキャブラリーへのトークンの登録と最大トークン数を学習
    train=Falseで学習は行わず既存のボキャブラリーからinput_idsの作成

    limit_length=int:トークン数に制限を設ける場合の指定数
    """
    
    if train: # 学習データとテストデータを別個に変換する場合
      self._training_vocab(data)
    else:
      loader = False 
    inputs = self._create_inputs_ids(data, limit_length)
    self._assert_test(data, inputs, limit_length)

    if loader:
      train_loader, val_loader, test_loader = self._loader(inputs, batch_size)
      return train_loader, val_loader, test_loader
    else:
      return inputs  

  def _loader(self, inputs, batch_size: int):
    # DataLoaderの作成
    import random 
    inputs = random.sample(inputs, len(inputs))
    # データ分割
    n_ = len(inputs)
    n_train = int(n_*.6)
    n_val = int(n_*.2)

    train = inputs[:n_train]
    val = inputs[n_train:n_train+n_val]
    test = inputs[n_train+n_val:]
    # バッチ分割
#     train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
#     val_loader = DataLoader(val, batch_size=batch_size)
#     test_loader = DataLoader(test, batch_size=batch_size)
    
    train_loader = MakeDataLoader(batch_size=batch_size, shuffle=True).my_word_loader(train)
    val_loader = MakeDataLoader(batch_size=batch_size).my_word_loader(val)
    test_loader = MakeDataLoader(batch_size=batch_size).my_word_loader(test)

    return train_loader, val_loader, test_loader 

class EnWordToTensor(WordToTensorLoader):

  def _make_wakati(self, text: str) -> list:
    # space 単位で分割
    return text.strip().split()

  def _preprocessing(self, text: str) -> str:
    text = PreProcessingTEXT()._en_preprocessinf(text)
    return text 

class JpWordToTensor(WordToTensorLoader):
    
  def __init__(self):
    self.tokenizer = Tokenizer()

  def _make_wakati(self, text: str) -> list:
    # 形態素による分割
    return [tok for tok in self.tokenizer.tokenize(text, wakati=True)]

  def _preprocessing(self, text: str) -> str:
    text = PreProcessingTEXT()._jp_preprocessing(text)
    return text 
