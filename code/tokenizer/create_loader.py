from torch.utils.data import DataLoader 
from tokenize_by_bert import BertJpTokenizer, BertEnTokenizer
from preprocessing import PreProcessingTEXT
import torch 

class MakeDataLoader(DataLoader):
  def __init__(self, batch_size: int=32, shuffle: bool=True, language: str="jp"):
    self.batch_size = batch_size
    self.shuffle = shuffle 
    self.language = language
    if self.language == "jp":
      self.tokenizer = BertJpTokenizer()
    elif self.language == "en":
      self.tokenizer = BertEnTokenizer()

  def my_word_loader(self, dataset):
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

  def bert_loader(self,
                  text_list: list,
                  labels_list: list,
                  padding: str="max_length", 
                  max_length: int=256,
                  truncation: bool=True,
                  return_tensor: bool=True, 
                  cuda: bool=False
                  ):
    data = []
    for text, label in zip(text_list, labels_list):
      if self.language == "jp":
        text = PreProcessingTEXT()._jp_preprocessing(text)
      elif self.language == "en":
        text = PreProcessingTEXT()._en_preprocessing(text)
      encoding = self.tokenizer(text,
                           padding=padding,
                           max_length=max_length,
                           truncation=truncation,
                           return_tensor=return_tensor,
                           cuda=cuda,
                           labels=label)
      data.append(encoding)

    return DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle)
