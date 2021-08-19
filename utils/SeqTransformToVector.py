from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np 
from pprint import pprint 
class CountVectorizer_:
    def __init__(self):
        self.model = None 
        print("[]")
    def _tokenizer_idently(self, sentence):
        return sentence 
    def fit_transform(self, sentence: list) -> list:
        self.model = CountVectorizer(lowercase=False, tokenizer=self._tokenizer_idently)
        return self.model.fit_transform(sentence).toarray()
    def fit(self, lowercase: bool):
        self.model = CountVectorizer(lowercase=lowercase, tokenizer=self._tokenizer_idently)
        print(self.model)
    def transform(self, sentence: list) -> list:
        if self.model != None:
            return self.model.fit_transform(sentence).toarray()
        else:
            raise NotImplementedError
    def vocabulary(self):
        if self.model != None:
            return self.model.vocabulary_
        else:
            raise NotImplementedError
    def inverse_transform(self, vector: list) -> list:
        """
        文章ベクトルから辞書によって単語を出力します
        """
        if self.model != None:
            return self.model.inverse_transform(vector)
        else:
            raise NotImplementedError

class TfidfVectorizer_(CountVectorizer_):
    def fit(self, lowercase: bool):
        self.model = TfidfVectorizer(lowercase=lowercase, tokenizer=self._tokenizer_idently)
        print(self.model)
    def transform(self, sentence: list) -> list:
        if self.model != None:
            return self.model.fit_transform(sentence).toarray()
        else:
            raise NotImplementedError
    def fit_transform(self, sentence: list) -> list: # return (seqence, word)
        self.model = TfidfVectorizer(lowercase=False,tokenizer=self._tokenizer_idently)
        return self.model.fit_transform(sentence).toarray()

class Doc2Vec_:
    def __init__(self, vector_size=128, window=5, min_count=1, worker=2, seed=888):
        self.model = None 
        self.param = dict(
            vector_size=vector_size,
            window=window,
            min_count= min_count,
            worker=worker,
            seed = seed 
        )
        pprint(self.param)
    def fit_wiki(self):
        """
        wikipediaで学習させた汎用的なモデルの読み込み
        ファイルのダウンロードが必要です
        """
        self.model = Doc2Vec.load("jawiki.doc2vec.dbow300d.model")
        print(self.model)

    def fit(self, sentence):
        """
        独自のデータ文章で学習させる
        """
        tagged_docs = [TaggedDocument(doc, [idx, f"doc-{idx:02d}"]) for idx, doc in enumerate(sentence)]
        self.model = Doc2Vec(tagged_docs, **self.param)
        print(self.model)
    def transform_sentence(self, sentence):
        """
        全文章に対してベクトル変換します
        """
        if self.model != None:
            vector = []
            for doc in sentence:
                vector.append(self.model.infer_vector(doc))
            return vector
    def transform_sequence(self, seq):
        """
        全文章中のある一文章に対して、ベクトル変換します
        """
        if self.model != None:
            return self.model.infer_vector(seq)
        else:
            raise NotImplementedError
    def predict(self, doc: list):
        """
        未知の文章に対してベクトルの予測変換します
        """
        if self.model != None:
            return self.model.infer_vector(doc)
        else:
            raise NotImplementedError
    def most_similar(self, id=1):
        """
        学習させた文章間の類似度を返します
        """
        sim = self.model.docvecs.most_similar(id)
        print(sim)

    def augment_by_normal_noise(self, seq: list, scale_rate=0.1) -> list:
        """
        指定された文章ベクトルに微小なノイズを加えてベクトルを作成します
        scale_rateを変化させることでノイズの分散を変更します
        """
        v = self.model.infer_vector(seq)
        noise = np.random.normal(0.0, scale=scale_rate*v.std(), size=v.shape)
        cos = cosine_similarity([v], [noise+v])
        print(cos)
        return noise+v