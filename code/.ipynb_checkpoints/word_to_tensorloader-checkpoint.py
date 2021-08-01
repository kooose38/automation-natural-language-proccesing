import re 
import torch 
from torch.utils.data import DataLoader 
from janome.tokenizer import Tokenizer

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
      self.vocab_size = len(vocab_)
      self.word2index = word2index
      self.max_len = max_len + 1

  def _create_inputs_ids(self, data: list):
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
      if self.max_len >= len(dummy):
        for i in range(self.max_len - len(dummy)):
          dummy.insert(0, 0) # この実装は文章の先頭からpaddingで埋めている
      else:
        dummy = dummy[:self.max_len]
      dummy.append(self.word2index["<cls>"]) # 末端に<cls>の追加
      sentence2tensor["input_ids"] = torch.tensor(dummy)
      sentence2tensor["labels"] = label2tensor.item()
      inputs.append(sentence2tensor)
    return inputs 

  def transform(self, data: list, loader: bool=True, batch_size: int=32, train: bool=True):
    """
    data = [
      {"text": "", "label": 1},
      {"text": "", "label": 0},
      ...
    ]

    loader=Trueでtrain, val, test のLoaderを作成。戻り値は３つ
    loader=Falseでdata単独でinput_idsの作成。戻り値は１つ

    train=Trueでボキャブラリーへのトークンの登録と最大トークン数を学習
    train=Falseで学習は行わず既存のボキャブラリーからinput_idsの作成
    """
    if train: # 学習データとテストデータを別個に変換する場合
      self._training_vocab(data)
    else:
      loader = False 
    inputs = self._create_inputs_ids(data)
    # テストコード
    assert inputs[0]["input_ids"].size()[0] == inputs[1]["input_ids"].size()[0]
    # DataLoaderの作成
    if loader:
      train_loader, val_loader, test_loader = self._loader(inputs, batch_size)
      return train_loader, val_loader, test_loader
    else:
      return inputs  

  def _loader(self, inputs, batch_size: int):
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
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, val_loader, test_loader 

class EnWordToTensor(WordToTensorLoader):

  def _make_wakati(self, text: str) -> list:
    # space 単位で分割
    return text.strip().split()

  def _preprocessing(self, text: str) -> str:
    text = re.sub("\n", " ", text)
    text = re.sub("\r", "", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text 

class JpWordToTensor(WordToTensorLoader):
    
  def __init__(self):
    self.tokenizer = Tokenizer()

  def _make_wakati(self, text: str) -> list:
    # 形態素による分割
    return [tok for tok in self.tokenizer.tokenize(text, wakati=True)]

  def _preprocessing(self, text: str) -> str:
    text = re.sub("\n", "", text)
    text = re.sub("\r", "", text)
    text = re.sub(" ", "", text)
    text = re.sub("　", "", text)
    text = re.sub(r'[0-9 ０-９]', '0', text) 
    return text 