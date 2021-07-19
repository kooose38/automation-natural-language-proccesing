from sklearn.preprocessing import OneHotEncoder
import math 
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

class OneHotEncoder_:
    def __init__(self):
        self.onehot = None 
    def fit_transform(self, sentence: list) -> list: # 形態素解析の形状で渡す
        word = [[word] for doc in sentence for word in doc]
        self.onehot = OneHotEncoder(handle_unknown="ignore")
        vector = self.onehot.fit_transform(word).toarray()

        return vector 
    def category(self):
        if self.onehot != None:
            return self.onehot.categories_[0]
        else:
            raise NotImplementedError

    def inverse_transform(self, vector):
        if self.onehot != None:
            return self.onehot.inverse_transform([vector]).ravel()
        else:
            raise NotImplementedError

class Word2Vec_:
    def __init__(self, vocab_size=128, min_count=1, window=5, workers=1, seed=123):
        self.model = None 
        self.vocab_size = vocab_size 
        self.min_count = min_count 
        self.window = window 
        self.workers = workers 
        self.seed = seed

    def _hash(self, s):
        n = len(s) 
        h = 0
        for idx, c in enumerate(s):
            h += ord(c) * math.factorial(n-idx)
        return h

    def fit(self, sentence: list):
        self.model = Word2Vec(sentence, 
                              size=self.vocab_size,
                              min_count=self.min_count,
                              window=self.window,
                              seed=self.seed,
                              hashfxn=self._hash)
        print(self.model)

    def transform_word(self, word: str):
        """
        Returns a vector on a word-by-word basis
        """
        if self.model != None:
            return ([word], self.model.wv[word])
        else:
            raise NotImplementedError

    def transfrom_sentence(self, sentence: list):
        """
        Returns a vector for every sentence
        """
        if self.model != None:
            new_sentence = []
            for doc in sentence:
                new_doc = []
                for line in doc:
                    new_line = self.model.wv[line]
                    new_doc.append(new_line)
                new_sentence.append(new_doc)
            return new_sentence
        else:
            raise NotImplementedError

    def similar_by_vector(self, vector, topn=4):
        if self.model != None:
            return self.model.wv.similar_by_vector(vector, topn=topn)
        else:
            raise NotImplementedError
    def similar_by_word(self, word:str, topn=4):
        if self.model != None:
            return self.model.wv.similar_by_word(word, topn=topn)
        else:
            raise NotImplementedError
    def vocabulary_(self):
        return self.model.wv.vocab.keys()

    def augment_by_normal_noise(self, word: str, scale_rate=0.1) -> list:
        """
        指定された単語ベクトルに微小なノイズを加えてベクトルを作成します
        scale_rateを変化させることでノイズの分散を変更します
        """
        v = self.model.wv[word]
        noise = np.random.normal(0.0, scale=scale_rate*v.std(), size=v.shape)
        cos = cosine_similarity([v], [noise+v])
        print(f"Cosine Similar: {cos}")
        print(f"restored_word1: {self.similar_by_vector(v, topn=1)}")
        print(f"restored_word1: {self.similar_by_vector(noise+v, topn=1)}")
        return noise+v