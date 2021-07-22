# !pip install transformers==4.5.0 fugashi==1.1.0 ipadic==1.0.0 pytorch-lightning==1.2.7
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl
class BertForSequenceClassificationMultiLabel(torch.nn.Module):
    
    def __init__(self, num_labels, model_name):
        """
        Bertによるマルチ分類/マルチラベルのタスクを行います。
        ここでの入出力の関係性を説明します。なお文章数はバッチサイズと同義です。
          (文章数, 単語数, 1) 入力層
            -> (文章数, 単語数, 768) BertModelによる１単語を768次元の（単語埋め込み）を出力します
              -> (文章数, 768) [PAD](空文字)を除いたトークンの平均を文章単位で取ります/attention-maskで判断します
                -> (文章数, class) 線形結合による次元削減を行います。
                  -> (文章数, class) sigmoidによる確率ベースに変換します。
        """
        super().__init__()
        # BertModelのロード
        self.bert = BertModel.from_pretrained(model_name) 
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, num_labels)
        )

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        labels=None
    ):
        # データを入力しBERTの最終層の出力を得る。
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids) # (32, 128, 1) -> (32, 128, 768)
        last_hidden_state = bert_output.last_hidden_state
        
        # [PAD]以外のトークンで隠れ状態の平均をとる
        averaged_hidden_state = \
            (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
            / attention_mask.sum(1, keepdim=True) # (32, 128, 768) -> (32, 768)
        
        # 線形変換
        scores = self.fc(averaged_hidden_state)  # (32, 768) -> (32, 3)
        # 出力の形式を整える。
        output = {'logits': scores}

        # labelsが入力に含まれていたら、損失を計算し出力する。
        if labels != None: 
            loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float())
            output['loss'] = loss
        # 属性でアクセスできるようにする。
        output = type('bert_output', (object,), output) 
        return output # logits(出力層), lossを指定できるオブジェクト

class BertForSequenceClassificationMultiLabel_pl(pl.LightningModule):

    def __init__(self, num_labels, model_name, lr):
        super().__init__()
        self.save_hyperparameters() 
        # (32, 128, 1) -> (32, 128, 768) -> (32, 766) -> (32, 3)
        self.bert_scml = BertForSequenceClassificationMultiLabel(
            num_labels, model_name
        ) 

    def training_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')  # lossを計算しないので正解ラベルの除去
        output = self.bert_scml(**batch)
        scores = output.logits
        labels_predicted = ( scores > 0 ).int() #[[1. 0. 0], [0, 1, 1] ...] 閾値が正負
        num_correct = ( labels_predicted == labels ).all(-1).sum().item() # 文章単位ですべて正解していたら
        accuracy = num_correct/scores.size(0)
        self.log('accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



class BertClassficationMultimodel:
  def __init__(self, model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
    self.model_name = model_name
    self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
    self.model = None
    self.best_model_path = "" # 学習時に最も検証スコアの高かった重みを保存するパス
    self.i = 0 # 推論時に何回目かのフラグ
    self.max_length = 0 # トークン数。推論時にもデータサイズをそろえたいのでここで保存している

  def _show_tensorboard(self):
    %load_ext tensorboard
    %tensorboard --logdir ./

  def _transform(self, text_list: list, labels: list, cuda: bool, batch_size: int):
    dataset = []
    # bertの入力データ作成
    for text, label in zip(text_list, labels):
      encoding = self.tokenizer(
          text,
          max_length=self.max_length, # 文章の長さに応じて要変更する
          padding="max_length",
          truncation=True
      )
      encoding["labels"] = label
      encoding = {k: torch.tensor(v) for k, v in encoding.items()}
      if cuda:
        encoding = {k: v.cuda() for k, v in encoding.items()}
      dataset.append(encoding)
    # 訓練、検証、テスト分割
    n = len(dataset)
    n_train = int(n*.6)
    n_val = int(n*.2)
    dataset_train = dataset[:n_train]
    dataset_val = dataset[n_train:n_train+n_val]
    dataset_test = dataset[n_train+n_val:]
    # バッチ単位で分割 (32, 128, 1) //32文章の塊で、１文章に128の単語、単語それぞれのトークン数1つ
    train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset_val, batch_size=256)
    test = DataLoader(dataset_test, batch_size=256)

    return train, val, test

  def _trainer(self, cuda, epoch: int):
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        dirpath="model/"
    )
    trainer = pl.Trainer(
        max_epochs=epoch,
        callbacks=[checkpoint]
    )
    if cuda:
      trainer["gpus"] = 1 
    return checkpoint, trainer
  def _test(self, trainer, test):
    test = trainer.test(test_dataloaders=test)
    print(f'テストデータ正解率: {test[0]["accuracy"]:.2f}')

  def train(self, text_list: list, labels: list, cuda=False, batch_size=32, lr=0.001, epoch=10, max_length=128):
    """
    ここでデータ分割、モデルの学習、テストデータの評価、モデルの保存を自動化します。
    入力データのサイズに注意してください。
    
    ### example 
        + text_list: [
              "sequence1",
              "sequence2",
              "sequence3",
              "sequence4",
            ]
        + labels: [
              [0, 1, 1], # この場合はクラス１,２にも属しているがクラス０には属していない/モデルの出力層は３つ
              [0, 1, 1],
              [1, 1, 1],
              [1, 0, 0],
            ]
        + cuda: gpuを使用するかbooleanで指定します
        + batch_size: 訓練データの学習時のバッチ数を指定します
        + lr: learning rate
        + epoch: 
        + max_lenght: トークン数
    """
    self.max_length = max_length
    train, val, test = self._transform(text_list, labels, cuda, batch_size)
    self.model = BertForSequenceClassificationMultiLabel_pl(len(labels[0]), self.model_name, lr)
    if cuda:
      self.model = self.model.cuda()
    # 学習
    checkpoint, trainer = self._trainer(cuda, epoch)
    trainer.fit(self.model, train, val)
    # 学習で最適だった重みの取得: string
    self.best_model_path = checkpoint.best_model_path 
    print('ベストモデルのファイル: ', checkpoint.best_model_path)
    print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)
    self._test(trainer, test)
    self._show_tensorboard()

  def _load_model(self):
    if self.best_model_path != "":
      # PyTorch Lightningモデルのロード
      model = BertForSequenceClassificationMultiLabel_pl.load_from_checkpoint(
          self.best_model_path
      )
      self.model = model.bert_scml 
    else: 
      raise NotImplementedError

  def predict(self, texts):
    self.i += 1
    if self.i == 1:
      self._load_model()
    enc = self.tokenizer(
        texts,
        max_length=self.max_length,
        padding="max_length",
        return_tensors="pt"
    )
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
      output = self.model(**enc)
      p_scores = sigmoid(output)
    scores = output.logits # (len(texts), num_class)
    labels_predicted = ( scores > 0 ).int().cpu().numpy().tolist()

    if type(texts) != list: # list型と同様にfor文でループさせるため
      texts = [texts]
      p_scores = [p_scores]
      labels_predicted = [labels_predicted]

    for text, pred, score in zip(texts, labels_predicted, p_scores):
      print("--")
      print(f"文章: {text}")
      print(f"予測ラベル: {pred}")
      print(f"確率: {score}")

model = BertClassficationMultimodel()
model.train(text_list, labels_list, batch_size=2)