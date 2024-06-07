import base64
import os

import openai
from dotenv import load_dotenv


# 画像をbase64にエンコードする関数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 画像のパス
image_path = "/Users/ailab/programming/separate_garbage/garbage_water bottle.png"

# 画像をbase64にエンコードする
base64_image = encode_image(image_path)

env = load_dotenv()
api_key=os.environ.get("OPENAI_API_KEY")

# チャットの応答を生成する
response = openai.ChatCompletion.create(
    # model の名前は gpt-4-vision-preview.
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "これのゴミの種類を判定してください．仮定でもいいので一つの単語で返してください"},  # ここに質問を書く
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},  # 画像の指定の仕方がちょい複雑
            ],
        }
    ],
    api_key=api_key,
    max_tokens=300,
)

# 応答を表示する
print(response.choices[0]["message"]["content"])

#参考文献
#https://qiita.com/kenji-kondo/items/87e71bf9645338d59ecb
