from collections import Counter
from itertools import chain

class AffixoidReference:
    affixoids = {
        # 呂淑湘 漢語語法分析問題
        "LuSX": [
            "可好難準類亞次超半單多部天非反自前代", # start            
            "員家人民界物品具件子種類別度率法學體質力氣性化" # end
        ],
        # 陳志偉等 漢語的構詞法
        "ChenZW": [
            "",
            "者家化個拉騰巴來然乎"
        ],
        # 任學良 漢語造詞法
        "RenXL": [
            "準二以",
            "論度式卻豪"
        ],
        #趙元任 漢語口語語法
        "ZhouYR": [
            "不單多泛準偽無非親反",
            "化的姓論觀率法界炎學家負"
        ],
        # 馬慶株 現代漢語詞綴的性質、範圍和分類
        "MaQZ": [
            "分準總偽亞可不",            
            list("家騰頭里當麼你們兒輛匹口張間本支點件"
             "篇上下個地面價為棍家界迷師戶化論派手"
             "性學員犯夫手兒壇體線子自哧嗒咕以悠棒"
             "當道溜氣實") + (
             "巴巴,沈沈,沖沖,嘟嘟,墩墩,光光,呼呼,"
             "乎兒乎兒,晃晃,撅撅,辣辣,溜溜,蒙蒙,囔囔,"
             "撲撲,茸茸,森森,生生,絲絲,騰騰,盈盈,"
             "油油,悠悠,滋滋,不唧").split(",")             
        ],
        #劉月華等 實用現代漢語語法
        "LiuYH":[
            "",
            "員長士家師生工匠手學論機器儀型形式度性則廠"
            "場站法"
        ],
        # 湯志祥 當代漢語詞語的共時狀態及其嬗變
        "TangZX": [
            "可反非誰"  # 準詞綴
            "超多高性軟核半全",  # 類詞綴
            "者員家士師生手夫星派鬼棍品性化然感壇" # 準詞綴
            "族盲戶學論觀界星機賽節式型群物劑金法" # 準詞綴
            "款庫服價罪犯案級牌片所率史亞度"  # 準詞綴
            "風熱難人車卡站票券水稅鞋肉舞歌班" # 類詞綴
        ],
        # 陳光磊 漢語詞法論
        "ChenGL": [
            "半超次打大單反泛非好可累前全偽小亞有準總",
            "夫家匠師生士員長手漢翁倌工星迷族佬鬼"
            "棍蛋重觀論學派界度率氣類品種件具子化"
            "性法是來角型式牌號熱業科處局廳店部組"
        ],
        # 郭良夫 現代漢語的前綴和後綴
        "GuoLF": [
            "多半制單超非無不反親自次",
            "品"
        ],
        # 沈孟瓔 漢語新的詞綴化傾向, 再彈漢語新的詞綴化傾向
        "ShenMY": [
            "可非無反多軟大高",
            "家員性化然手貧者學度型熱戶感族盲爺壇"
        ]
    }


    terms = {# 呂淑湘 漢語語法分析問題
        "LuSX": "類前綴、類後綴",       
        # 陳志偉等 漢語的構詞法
        "ChenZW": "類乎後置成分的東西",
        # 任學良 漢語造詞法
        "RenXL": "準詞頭、準詞尾",
        #趙元任 漢語口語語法
        "ZhouYR": "類詞綴",
        # 馬慶株 現代漢語詞綴的性質、範圍和分類
        "MaQZ": "類詞綴",
        #劉月華等 實用現代漢語語法
        "LiuYH": "類後綴",
        # 湯志祥 當代漢語詞語的共時狀態及其嬗變
        "TangZX": "準詞綴、類詞綴",
        # 陳光磊 漢語詞法論
        "ChenGL": "類詞綴",
        # 郭良夫 現代漢語的前綴和後綴
        "GuoLF": "新興的前綴、後綴",
        # 沈孟瓔 漢語新的詞綴化傾向, 再彈漢語新的詞綴化傾向
        "ShenMY": "詞綴化傾向和進程、類詞綴、準詞綴"

    }
    
    def __init__(self):
        data = AffixoidReference.affixoids.values()
        self.__prefixes = list(chain.from_iterable(list(x[0]) for x in data))
        self.__suffixes = list(chain.from_iterable(list(x[1]) for x in data))
    
    def get_prefixes(self):
        return list(set(self.__prefixes))
    
    def get_suffixes(self):
        return list(set(self.__suffixes))

    def get_prefix_nominations(self):
        return Counter(self.__prefixes)
    
    def get_suffix_nominations(self):
        return Counter(self.__suffixes)
