# !pip install -q ipadic transformers fugashi 
from transformers import BertTokenizer, BertJapaneseTokenizer 
from typing import Any 
from bert_model_name import ModelNameByBertTokenizer

class BasicToken:

  def tokenize(self, text: str) -> list:
    # separate sentence 
    return self.tokenizer.tokenize(text)

  def encode(self, text: str) -> list:
    # return input_ids only 
    return self.tokenizer.encode(text)

  def convert_ids_to_tokens(self, input_ids: list) -> list:
    return self.tokenizer.convert_ids_to_tokens(input_ids)

  def convert_tokens_to_ids(self, text: str) -> list:
    return self.tokenizer.convert_tokens_to_ids(text)

  def __call__(self, text: Any,
               padding: str="",
               max_length: int=0,
               truncation: bool=False,
               return_tensor: bool=True,
               cuda: bool=False,
               labels: Any=None,
               ):
    # return input_ids , attention_mask and token_type_ids 
    param = {
        "text": text,
    }

    if padding != "":
      param["padding"] = padding # ["max_length", "longest"]
    if max_length != 0:
      param["max_length"] = max_length
    if truncation:
      param["truncation"] = truncation

    tokenized = self.tokenizer(**param)

    if labels != None:
      tokenized["labels"] = labels
    if cuda:
      tokenized = {k: v.cuda() for k, v in tokenized.items()}
    if return_tensor:
      tokenized = {k: torch.tensor(v, dtype=torch.long) for k, v in tokenized.items()}

    return tokenized

class BertJpTokenizer(BasicToken):
  def __init__(self):
    self.model_name = ModelNameByBertTokenizer().jp_model_name 
    self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)

class BertEnTokenizer(BasicToken):
  def __init__(self):
    self.model_name = ModelNameByBertTokenizer().en_model_name 
    self.tokenizer = BertTokenizer(vocab_file=self.model_name, do_lower_case=True)
    self.vocab_size = 0 
    self._get_vocab()

  def _get_vocab(self):
    word = {}
    with open(self.model_name, "r") as f:
      data = f.read().strip().split("\n")
      for d in data:
        if d not in word:
          word[d] = len(word)
    self.vocab_size = len(word)