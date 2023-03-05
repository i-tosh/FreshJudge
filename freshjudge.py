import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from PIL import Image
from torchvision import transforms, datasets

import random
import os
import glob

from torchvision.models import resnet18

st.set_page_config(layout="wide", page_title="ナス鮮度判定")

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 5)
        self.batch_size = 8

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            accuracy(y.softmax(dim=-1), t),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_acc", accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test_acc", accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


# ネットワークの準備
net = Net()

net.eval()  # 推論モード
# 重みの読み込み
net.load_state_dict(torch.load("eggplnt.pt"))




# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

recp = ["麻婆ナス", "ナスのたたき", "ナスの煮物", "焼きナス"]

Foodstuff0 = {"豚ひき肉": "250g", "ナス": "4本", "ピーマン": "2個", "長ネギ": "1/2本"}
Foodstuff1 = {
    "なす": "3本",
    "みょうが": "3個",
    "大葉": "10枚",
    "しらす干し": "大さじ4",
    "ゆずポン酢": "大さじ5",
    "ごま油": "小さじ1",
}
Foodstuff2 = {"なす": "５本", "サラダ油": "大さじ４", "濃口しょう油": "大さじ１", "さとう": "大さじ２"}
Foodstuff3 = {
    "なす": "４本",
    "◆醤油": "大さじ４",
    "◆砂糖": "大さじ２",
    "◆すりゴマ": "大さじ２",
    "◆生姜": "一かけ",
    "オリーブ油（ごま油でもo.k)": "適量",
}


Foodstuff = [Foodstuff0, Foodstuff1, Foodstuff2, Foodstuff3]

recipe0 = [
    "茄子は揚げるか多めの油で炒め揚げる感じで火を通し、一度取り出しておきます。 にんにく、しょうがを炒め香りが出たらひき肉を入れ炒めます。崩さなくてもそのままの塊で焼き色をつけます。ほぼ火が通ったら崩して炒めます。",
    "あらかじめＡを混ぜておいた調味料を②に入れます。すでに水溶き片栗粉が入ってるので弱火にします。",
    "ゆっくりとあんになっていきます。ほぼあんになったら①の茄子をここで戻し入れ、一緒に細かく切った長ネギも入れます。",
    "最後、香りづけにごま油をひとまわりかけます。※茄子を入れたらざっくり混ぜてね！",
]
recipe1 = [
    "なすは洗って、縦半分に切り、５ミリくらいの厚さに切る。",
    "耐熱容器に並べ、ラップをふんわりかけ、600ｗで5分加熱する。",
    "加熱している間に、みょうがは縦半分に切り、千切りにする。大葉も千切りにする",
    "ゆずポン酢とごま油は合わせておく。",
    "２の加熱が終わったら皿に盛り、熱いうちに４のゆずポン酢をかける。",
    "みょうが･大葉･しらす干しをのせたら完成です。",
]
recipe2 = [
    "ナスは大きめの乱切りにする。水気をよく拭いておく。",
    "鍋にサラダ油を入れ焦げ目をつけないようにナスを炒める。",
    "茄子に油がまわったら、ひたひたに水を入れて砂糖と醤油を加えて中火で煮る。焦がさないように時々火加減をみて煮詰めていく。",
    "ナスが柔らかくなり、煮汁がほとんどなくなって照りが出てきたら完成☆冷めたころがおいしい♪",
]
recipe3 = [
    "なすは縦に４当分して水に浸してあくを抜きます。",
    "ボールに◆をあわせておきます。",
    "１のなすを水切りし、フライパンにオリーブ油を少し多めに入れなすを両面こんがり焼きます。",
    "焼きあがったなすから順にボールの調味料に潜らせてお皿に盛ります。残った調味料を上からかけて出来上がり！" "",
]

recipe = [recipe0, recipe1, recipe2, recipe3]

rcp_url0 = "templates/img/mabo.jpg"
rcp_url1 = "templates/img/tataki.jpg"
rcp_url2 = "templates/img/nimono.jpg"
rcp_url3 = "templates/img/yaki.jpg"
rcp_url = [rcp_url0, rcp_url1, rcp_url2, rcp_url3]

RCP_NUM = 4  # レシピ数

# タイトルとテキストを記入
st.title('ナスの鮮度判定')
st.write('鮮度１(腐敗)<-->鮮度５(新鮮)')

# 画像upload
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    image = Image.open(my_upload)
    st.image(image)

    n = random.randrange(RCP_NUM)  # レシピをランダムに選択する

    img2 = transform(image).unsqueeze(0)
#n徝は暫定
#    n=0
    context = [
        {"name": "新鮮度1", "discrpt": "即処分しましょう" },
        {"name": "新鮮度2", "discrpt": "チャレンジ精神があるなら食べられる"},
        {
            "name": "新鮮度3",
            "discrpt": "早めに食べましょう",
            "recipe": recp[n],
            "recp_pic": rcp_url[n],
            "foodstuff": Foodstuff[n],
            "rcp": recipe[n],
        },
        {
            "name": "新鮮度4",
            "discrpt": "今のうちに食べちゃいましょう",
            "recipe": recp[n],
            "recp_pic": rcp_url[n],
            "foodstuff": Foodstuff[n],
            "rcp": recipe[n],
        },
        {
            "name": "新鮮度5",
            "discrpt": "最高に新鮮です",
            "recipe": recp[n],
            "recp_pic": rcp_url[n],
            "foodstuff": Foodstuff[n],
            "rcp": recipe[n],
        },
    ]

    # 画像識別
    y = net(img2)
    han = F.softmax(y)
    acc, pred = torch.topk(han, 1)
    #            pred = torch.argmax(han)
    #            acc = han[n] * 100
    acc = 100 * acc[0].detach().numpy()
    #context[pred]["acc"] = acc
    #st.write(context[pred])
    text = context[pred]
    
    st.subheader(text["name"])
    st.subheader(text['discrpt'])
    if(pred>1):
        st.subheader(text['recipe'])
        
        pict = Image.open(text['recp_pic'])
        st.image(pict)
        
        st.subheader("食材")
        text['foodstuff']

        st.subheader("レシピ")

        text['rcp']