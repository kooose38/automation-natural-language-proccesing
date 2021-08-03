import re 

class PreProcessingTEXT:
  
  def __init__(self):
    pass

  def _en_preprocessing(self, text: str) -> str:
    text = re.sub("\n", " ", text)
    text = re.sub("\r", "", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text 

  def _jp_preprocessing(self, text: str) -> str:
    text = re.sub("\n", "", text)
    text = re.sub("\r", "", text)
    text = re.sub(" ", "", text)
    text = re.sub("　", "", text)
    text = re.sub(r'[0-9 ０-９]', '0', text) 
    return text 