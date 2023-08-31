# 참고 논문: Efficient K-NN for Playlist Continuation (RecSys'18 Challenge)
# 주소: https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf
# 참고 Notebook: 제목으로 태그 맞추기 with Khaiii, Colab
# 주소: https://arena.kakao.com/forum/topics/226
# 참고 Notebook: Melon Playlist Continuation 대회 데이터 전처리 & EDA
# 주소: https://arena.kakao.com/forum/topics/191


import numpy as np
import pandas as pd
#from khaiii import KhaiiiApi
from konlpy.tag import Mecab
import re

train = pd.read_json('/data/mountain/train.json')
test = pd.read_json('/data/mountain/val.json')
val = pd.read_json('/data/mountain/test.json')

def combine_tags(df):
    return ' '.join(df['tags'])

train['comb_tags'] = train.apply(combine_tags, axis=1)
val['comb_tags'] = val.apply(combine_tags, axis=1)
test['comb_tags'] = test.apply(combine_tags, axis=1)

def re_sub(series):
    series = series.str.replace(pat=r'_{1,}', repl=r' ', regex=True)
    series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
    series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
    series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
    series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
    return series

def get_token(title, tokenizer):
    
    if len(title)== 0 or title== ' ':  # 제목이 공백인 경우 tokenizer에러 발생
        return []
    
    result = tokenizer.pos(title)
    #result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
    return result

tokenizer = Mecab()

train['plylst_title'].fillna('', inplace=True)
val['plylst_title'].fillna('', inplace=True)

train['plylst_title'] = re_sub(train['plylst_title'])
train.loc[:, 'ply_token'] = train['plylst_title'].map(lambda x: get_token(x, tokenizer))
train['comb_tags'] = re_sub(train['comb_tags'])
train.loc[:, 'tag_token'] = train['comb_tags'].map(lambda x: get_token(x, tokenizer))

val['plylst_title'] = re_sub(val['plylst_title'])
val.loc[:, 'ply_token'] = val['plylst_title'].map(lambda x: get_token(x, tokenizer))
val['comb_tags'] = re_sub(val['comb_tags'])
val.loc[:, 'tag_token'] = val['comb_tags'].map(lambda x: get_token(x, tokenizer))

test['plylst_title'] = re_sub(test['plylst_title'])
test.loc[:, 'ply_token'] = test['plylst_title'].map(lambda x: get_token(x, tokenizer))
test['comb_tags'] = re_sub(test['comb_tags'])
test.loc[:, 'tag_token'] = test['comb_tags'].map(lambda x: get_token(x, tokenizer))

using_pos = ['NNG','SL','NNP','MAG','SN', 'NP', 'VA', 'VV', 'XR']
train['ply_token'] = train['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
train['tag_token'] = train['tag_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

val['ply_token'] = val['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
val['tag_token'] = val['tag_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

test['ply_token'] = test['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
test['tag_token'] = test['tag_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))

train['title_token'] = train['ply_token'].map(lambda x: [tag[0] for tag in x])
train['tags_token'] = train['tag_token'].map(lambda x: [tag[0] for tag in x])

val['title_token'] = val['ply_token'].map(lambda x: [tag[0] for tag in x])
val['tags_token'] = val['tag_token'].map(lambda x: [tag[0] for tag in x])

test['title_token'] = test['ply_token'].map(lambda x: [tag[0] for tag in x])
test['tags_token'] = test['tag_token'].map(lambda x: [tag[0] for tag in x])

train.drop(['ply_token', 'tag_token', 'comb_tags'], axis=1, inplace=True)
val.drop(['ply_token', 'tag_token', 'comb_tags'], axis=1, inplace=True)
test.drop(['ply_token', 'tag_token', 'comb_tags'], axis=1, inplace=True)

train.to_json('/home/q1cho/kakao_arena/거위/data/train_with_tt_token3.json', orient='table')
val.to_json('/home/q1cho/kakao_arena/거위/data/val_with_tt_token3.json', orient='table')
test.to_json('/home/q1cho/kakao_arena/거위/data/test_with_tt_token3.json', orient='table')
