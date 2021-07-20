import torch 
import torch.nn as nn 
import numpy as np 
import torch 
from transformers import BertJapaneseTokenizer, BertModel
class MyBertModel(nn.Module):
    def __init__(self,
                 num_word,
                 n_class,
                 cuda,
                 model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
        """
        bertModelによる単語の分散表現/bertModel+Linearの線形結合によるクラス分類を行います
        通常bertModelはトークンごとに768次元のベクトルに変換します。
        このモデルでは、bertを利用した転移学習を行います
        """
        super(MyBertModel, self).__init__()
        with torch.no_grad(): # ファインチューニングしたいなら、ここをコメントにしてください
           self.model = BertModel.from_pretrained(model_name)
        # 分類の一例なのでタスクに応じてモデルの形状を変えること
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768*num_word, 768), # トークン数*ベクトル次元数
            nn.ReLU(),
            nn.Linear(768, n_class),
        )
        if cuda:
            self.model = self.model.cuda()
            self.fc = self.fc.cuda()
        print(self.model.config)
    def forward(self, x):
        output = self.model(**x).last_hidden_state # (文, 単語, 1) -> (文, 単語, 768)
        output = self.fc(output) # 分類なら線形代数用いる -> (文, 10)
        return output 

# example learning code 
class BertModelClassfication:
    def __init__(self):
        self.bert = None
    def train(self, x, t: list, epoch=50, lr=0.001, n_class=10, cuda=False):
        t = torch.tensor(t)
        self.bert = MyBertModel(num_word=len(x["input_ids"][0]), n_class=n_class, cuda=cuda)
        optimzer = torch.optim.Adam(self.bert.parameters(), lr=lr)
        loss_f = nn.CrossEntropyLoss()

        self.bert.train()
        all_loss = 0
        for i in range(epoch+1):
          y = self.bert(enc) 
          loss = loss_f(y, t)
          optimzer.zero_grad()
          loss.backward()
          optimzer.step()

          all_loss += loss 
          if i%10 == 0:
            print(f"epoch:[{i} / {epoch}] loss:{loss}")
        print(f"avgloss by 1 epoch: {all_loss/epoch}")

    def metrics(self, x, t: list):
        self.bert.eval()
        with torch.no_grad():
          y = self.bert(x)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(y, torch.tensor(t))
        y = y.argmax(-1).cpu().numpy()
        t = np.array(t)
        correct = (t == y).sum()
        print(f"accuracy: {(correct/len(t))*100}% loss: {loss}")

    def predict(self, x):
        self.bert.eval()
        softmax = nn.Softmax()
        with torch.no_grad():
          y = self.bert(x)
          y = softmax(y)
        return y.argmax(-1).cpu().numpy().tolist()