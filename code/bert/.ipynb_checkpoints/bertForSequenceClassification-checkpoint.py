# !pip install transformers==4.5.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.2.7
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
### -----------------------------------------------分類モデル---------------------------------------------------------------------###
class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name: str, num_labels: int, lr: float):
        """
        bertを使ったファインチューニングによる他クラス分類/シングルラベルを行います。
        bert層では、
         (文章数, 単語数, 1) 1は単語におけるトークン
           -> (文章数, 単語数, 768) BertModelによる１単語を768次元のベクトル(単語埋め込み)で出力
             -> (文章数, 768) トークン[CLS]のみのベクトルを抽出 [CLS]には一文章ベクトルの特長量がまとまったものと同意
               -> (文章数, クラス数) 通常の線形結合による次元圧縮/sofmmaxによる確率表現 
        """
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        print(self.bert_sc.config)
        print("~"*100)
        
    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    # バッチ単位でloss値の計算
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch) # (32, 128, 1) -> (32, 128, 768) -> (32, 768) -> (32, 10)
        loss = output.loss # 入力データにlabelsを含めることでlossが取得できる。これはbert側での実装です。
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss
        
    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを除去して変数化
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1) # indexの取得
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
### --------------------------------------------実行環境（自動的）-----------------------------------------------------------###
class MyBertSequenceClassification:
  def __init__(self, model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
    self.model_name = model_name 
    self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
    self.model = None
    self.best_model_path = "" # 最適なモデルの保存パス
    self.i = 0 # 推論の回数
    self.max_length = 0 # 推論時に使うので保持

  def _show_tensorboard(self):
    %load_ext tensorboard
    %tensorboard --logdir ./
  def _load_model(self):
    if self.best_model_path != "":
      # PyTorch Lightningモデルのロード
      model = BertForSequenceClassification_pl.load_from_checkpoint(
          self.best_model_path
      ) 
      # Transformers対応のモデルを./model_transformesに保存
      model.bert_sc.save_pretrained('./model_transformers') 
      print("./model_transformers 学習済モデルの保存をしました。")
      self.model = None 
    else:
      raise NotImplementedError

  def _on_load_model(self):
    # Transfomersのモデルとして保存
    self.model = BertForSequenceClassification.from_pretrained(
        './model_transformers'
    )
  def _transform(self, text_list, labels, batch_size: int, cuda: bool):
    """
    それぞれの文章をトークン化します
    データから訓練、検証、テストに分割します
    バッチ単位でそれぞれを分けます。なお訓練以外のデータに対しては学習をしないのでバッチサイズは大きく設定してます。
    バッチサイズ == 文章数の単位

    (文章数, 1) -> (#バッチ, 1バッチの文章数, 1文章の単語数, 1単語1トークン) #わかりやすくするための表記であり、理論上存在しません
    """
    inputs = []
    for text, label in zip(text_list, labels):
      encoding = self.tokenizer(
          text,
          max_length=128, # 文章の長さに応じて要変更
          padding="max_length",
          truncation=True
      )
      self.max_length = 128
      encoding["labels"] = label 
      encoding = {k: torch.tensor(v) for k, v in encoding.items()}
      if cuda:
        encoding = {k: v.cuda() for k, v in encoding.items()}
      inputs.append(encoding)
    random.shuffle(inputs) # ランダムにシャッフル
    n = len(inputs)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = inputs[:n_train] # 学習データ
    dataset_val = inputs[n_train:n_train+n_val] # 検証データ
    dataset_test = inputs[n_train+n_val:] # テストデータ

    train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True
    )  # (32, 128, 1) * 元データの長さ/バッチ数
    val = DataLoader(dataset_val, batch_size=256)
    test = DataLoader(dataset_test, batch_size=256)
    return train, val, test 
  def _trainer(self, cuda: bool, epoch: int):
        """
        学習インスタンスの定義
        """
    checkpoint = pl.callbacks.ModelCheckpoint(
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      save_weights_only=True,
      dirpath='model/',
    )
    # 学習の方法を指定
    trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks = [checkpoint]
    )
    if cuda:
      trainer["gpus"] = 1
    return checkpoint, trainer

  def _test(self, test, trainer):
    test = trainer.test(test_dataloaders=test)
    print(f'テストデータ正解率: {test[0]["accuracy"]:.2f}')

  def train(self, text_list: list, labels: list, batch_size=32, cuda=False, epoch=10, lr=0.001):
    """
    モデルを使って学習、検証結果の最も高い重みの保存。テストデータの正解率を出力します。
    実行には多くの時間を要します
    引数の指定:
      test_list: ["1文章", "2文章", ...]
      labels: [0, 1, 1, 3, 2....]
      batch_size: 学習時のバッチ数
      cuda: gpuを使うか。ただしgpu環境が必要です。
      epoch: エポック数
      lr: 学習率
    """
    train, val, test = self._transform(text_list, labels, batch_size, cuda)
    checkpoint, trainer = self._trainer(cuda, epoch)
    self.model = BertForSequenceClassification_pl(self.model_name, len(set(labels)), lr=lr)
    if cuda:
      self.model = self.model.cuda()
    trainer.fit(self.model, train, val)
    best_model_path = checkpoint.best_model_path
    self.best_model_path = best_model_path
    print('ベストモデルのファイル: ', checkpoint.best_model_path)
    print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)
    self._test(test, trainer)
    self._show_tensorboard()

  def predict(self, text: str, category={}):
    """
    学習済モデルから推論をします。
    """
    self.i += 1 
    if self.i == 1: # 最初の一度のみモデルの読み込みをします
      self._load_model()
      self._on_load_model()
    enc = self.tokenizer(text,
                         max_length=self.max_length,
                         padding="max_length",
                         return_tensors="pt")
    softmax = torch.nn.Softmax()
    with torch.no_grad():
      output = self.model(**enc)
      scores = output.logits
      scores = softmax(scores)
    print(scores)
    predict = scores[0].argmax(-1).cpu().numpy()
    print(f"bertのモデルから、クラス{predict}と予測しました。")
    if len(category) != 0:
      print(f"カテゴリーは{category[predict]}です。")

# b = MyBertSequenceClassification()
# b.train(dataset_for_loader, labels)


### -----転移学習の例--------　
# import numpy as np
# class MyBertClassfication(torch.nn.Module):
#   def __init__(self, num_labels):
#     super(MyBertClassfication, self).__init__()
#     with torch.no_grad():
#       self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
#     self.fc = torch.nn.Sequential(
#         torch.nn.Linear(num_labels, num_labels),
#         torch.nn.ReLU()
#     )
#     self.soft = torch.nn.Softmax()
#   def forward(self, x):
#     output = self.model(**x)
#     output = output.logits
#     for _ in range(np.random.randint(0, 4, 1)[0]):
#       output = self.fc(output)
#     output = self.soft(output)
#     return output 


## ネガポジの例
# text_list = ["一位になれてうれしいです", "このパソコンは高すぎる", "今夜から病気にかかりました"]
# label_list = [1, 0, 0]
# epoch = 10
# my_bert = MyBertClassfication(2)
# optimizer = torch.optim.Adam(my_bert.parameters(), lr=0.001)
# loss_f = torch.nn.CrossEntropyLoss()
# e = tokenizer(text_list,
#               max_length=12,
#               padding="max_length", 
#               return_tensors="pt",
#               truncation=True)

# l = torch.tensor(label_list)
# losses = 0.0
# for i in range(epoch):
#   y = my_bert(e)
#   pred = y.argmax(-1)
#   loss = loss_f(y, l)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   losses += loss 
#   acc = (pred == l).sum()
#   print(f"batch: {i+1} -- Loss: {loss} -- acc: {acc/l.size()[0]}")