import lxml.html 
import requests 
import re 
import io 
import zipfile
import json 
from jq import jq # pip install jq==0.1.8
class LoadDataset:
    def __init__(self):
        self.url = None  

    def __doc__(self):
      """
      urlから処理を行い、文章単位で行リストを返します
      """
      pass 

    def _clern_text(self, lines: list):
      # クレンジング処理
      # 必要に応じて変更する
        prepped_lines = []
        for line in lines:
            line = line.strip()
            line = re.sub(r"\u3000", " ", line)
            line = re.sub(r" (.*?) ", "", line)

            line = re.sub(r"｜", "", line)
            line = re.sub(r"《.*?》", "", line)
            # 入力者の注釈を削除
            line = re.sub(r"※?［＃.*?］", "", line)
            # Unicode の全角スペースを削除
            line = re.sub(r"\u3000+", " ", line)
            if line == "":
                continue
            prepped_lines.append(line)

        return prepped_lines


    def load_zip(self, url: str) -> list:
        try:
            self.url = url 
            res = requests.get(self.url)
            z = zipfile.ZipFile(io.BytesIO(res.content))
            z.extractall("./data")

            txt_file = "./data/"+z.infolist()[0].filename

            with open(txt_file, "r", encoding="ShiftJIS") as f:
                lines = f.readlines()
            # 指定の文字列で区切る
            idx_start, idx_end = 0, len(lines)
            for idx, line in enumerate(lines):
                if re.search(r"^---", line):
                    idx_start = idx + 1

                if re.search(r"^底本", line):
                    idx_end = idx - 1 

            lines = lines[idx_start:idx_end]

            predded_lines = self._clern_text(lines)
            return predded_lines 

        except:
            print("error")
        
    def load_html(self, url: str) -> list:
        try:
            self.url = url 
            
            res = requests.get(self.url)
            if res.status_code != 200:
                raise Exception(f"HTTP Error. status code: {res.status_code}")
                
            html = lxml.html.fromstring(res.content)
            
            content = html.text_content().replace("\r\n", "\n")
            lines = content.split("\n") #行単位で分割

            prepped_lines = self._clern_text(lines)

            return prepped_lines 
        except:
            print("URLを入力してください。")

    def load_json(self, filePath: str) -> list:
        with open(filePath, "r") as f: 
            jsn = json.load(f)

        text = jq('.query.pages."68".extract').transform(jsn, text_output=True)
        print(text)
        formatted = re.sub(r"</?\w+(\s+[^>]+)?>", "", text)
        # `\n` という文字列を、改行コードに変更
        formatted = formatted.replace("\\n", "\n")
        formatted = formatted.split("\n")

        predded_lines = self._clern_text(formatted)
        return predded_lines
