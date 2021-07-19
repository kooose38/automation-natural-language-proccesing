# !pip install --upgrade google-cloud-translate
from google.colab import drive 
drive.mount("/content/gdrive")

import os 
os.environ["GOOGLE_APPLICATION_CREDENTIAL"] = "gdrive/My Drive/secret/key.json"

from google.cloud import translate_v2 as translate 

class TranslateFormGoogle:
    def __init__(self):
        self.translate_client = translate.Client()
    def augment_by_translate(self, doc: list, do_print=True) -> list:
        """
        google翻訳により文章を変換してデータの水増しを行います
        google colabで実行してください
        ただし有料です。
        """
        augment = []
        for text in doc:
            # 日から米
            translation = self.translate_client(
                text, target_language="en"
            )
            english = translation["translatatedText"]
              # 米から日
            translation = self.translate_client(
                english, target_language="en"
            )
            japanese = translation["translatatedText"]

            augment.append(japanese)
            if do_print:
                print("~"*70)
                print(f"Original: {text}")
                print(f"English: {english}")
                print(f"Japanese: {japanese}")
        return augment 