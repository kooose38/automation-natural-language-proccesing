import torch 
from transformers import BertJapaneseTokenizer, BertModel
class BasicBertTokenizer:
    def __init__(self, 
                 model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    def tokenize(self, text:str) -> list:
        """
        bertによる形態素解析を返します
        """
        return self.tokenizer.tokenize(text)
    def encode(self, text: str) -> list:
        """
        return tokenizer.input_ids
        """
        return self.tokenizer.encode(text)
    def tokenizer_(self,
                  text,
                  max_length=20, 
                  padding="max_length",
                  truncation=True,
                  cuda=False,
                 return_tensor=True):
        """
        bertModelに入力するデータ型に変換します。
        torch.Tensor/cuda
        """
        if type(text) == "str":
            encoding = self.tokenizer(text,
                                 max_length=max_length,
                                 padding=padding,
                                 truncation=truncation)
        else:
            encoding = self.tokenizer(text,
                                  padding=padding)
     
        if return_tensor:
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        if cuda:
            encoding = {k: v.cuda() for k, v in encoding.items()}
        return encoding
    def convert_ids_to_tokens_(self, input_ids: list) -> list:
        return self.tokenizer.convert_ids_to_tokens(input_ids) 