import MeCab
import janome.tokenizer 
import janome.analyzer
import janome.charfilter
import janome.tokenfilter
import nagisa 
# 汎用型クラス
class Transer(object):
    def transform(self, X, **kwargs):
        return X

    def fit(self, X, y, **kwargs):
        return self


class JpTokenizer(Transer):
    stop_poses = ["BOS/EOS", "助詞", "助動詞", "接続詞", "記号", "補助記号", "未知語"] # 除外する品詞

    def transform(self, X, **kwargs): # [[一行, 一行], [一行]]のテキストで渡す
        # 文書(行リスト)単位でループ
        new_docs = []
        for document in X:
            split_word = []
            # 行単位でループ
            for line in document:
                word = self._tokenize(line)
                split_word.extend(word)
            new_docs.append(split_word)
        return new_docs # [[word, word, word], [word, word...]]

    # 行単位のトークナイズ処理
    def _tokenize(self, line: str) -> list:
        # return line.split(" ") for example
        raise NotImplementedError("tokenize()")

# 以降、特化型クラス
# mecab
class JpTokenizerMeCab(JpTokenizer):
  
    def __init__(self):
        dicdir = ("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        self.taggerstr = f"-O chasen -d {dicdir}"

        self.tokenizer = MeCab.Tagger(self.taggerstr)

    # 行単位のトークナイズ処理(の実装)
    def _tokenize(self, line: str) -> list:
        # 行文字列を受け取りトークナイズを行う
        sentence = []
        parsed = self.tokenizer.parse(line)
        splitted = [l.split("\t") for l in parsed.split("\n")]
        for s in splitted:
            if len(s) == 1:     # may be "EOS"
                break
            surface, yomi, base, features = s[:4]
            word = surface      # surface form
            # word = base         # original form
            pos = features.split("-")[0]
            if pos not in self.stop_poses:
                sentence.append(word)
        return sentence

# janome
class JpTokenizerJanome(JpTokenizer):

    def __init__(self):
        tokenizer = janome.tokenizer.Tokenizer()
        token_filter = [janome.tokenfilter.POSStopFilter(self.stop_poses)]
        char_filter = [janome.charfilter.UnicodeNormalizeCharFilter()]
        self.aly = janome.analyzer.Analyzer(char_filters=char_filter,
                                             tokenizer=tokenizer,
                                             token_filters=token_filter)
    def _tokenize(self, line: str) -> list:
        sentence = []
        for token in self.aly.analyze(line):
            sentence.append(token.surface)

        return sentence  

# Nagisa
class JpTokenizerNagisa(JpTokenizer):
    def __init__(self):
        pass 
    def _tokenize(self, line: str) -> list:
        sentence = []
        parsed = nagisa.filter(line, filter_postags=self.stop_poses)

        for word, pos in zip(parsed.words, parsed.postags):
            sentence.append(word)

        return sentence 

  