import os
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
tqdm.pandas()

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

def rec_each_song(playlst_id, song_id, songs_len):
    # 전역변수 : sim_df, train, val, best300, train_songs
    seen_song = val[val['id'] == playlst_id]['songs'].values[0]
    # songs_len이 0이 아닌 경우
    if songs_len != 0: 
        n = 100//songs_len + 20 # 23       
        rec_lst = []
        # song_id가 sim_df에 있는 경우
        if song_id in sim_df.columns:
            series = sim_df[song_id].sort_values(ascending=False)
            series = series.drop(song_id)
            rec_lst = series.head(n).to_frame().index.tolist()
            return remove_seen(seen_song, rec_lst)      
        # song_id가 sim_df에 없는 경우
        else:
            if song_id in train_songs.values.tolist(): # song_id가 train에 있는 경우
                playlst_ids = train_songs[train_songs==song_id].index.tolist()
                dict_like = {}
                for i in playlst_ids:
                    dict_like[train.loc[i]['id']] = train.loc[i]['like_cnt']
                rec_lst = train[train['id'] == sorted(dict_like.items(), key=lambda x: x[1], reverse=True)[0][0]]['songs'].values[0] 
                if len(rec_lst) >= n: 
                    rec_list = rec_lst[:n] # rec_list : rec_lst에서 n개만큼 추출
                    return remove_seen(seen_song, rec_list) 
                else:
                    lst = remove_seen(rec_lst, best300) # best300에서 rec_lst를 제외한 리스트
                    rec_lst += lst[:n-len(rec_lst)] # rec_lst + lst에서 n-len(rec_lst)만큼 추출한 리스트 (총n개)
                    return remove_seen(seen_song, rec_lst) # rec_lst에서 seen_song을 제외한 리스트
            else: # song_id가 train에 없는 경우
                rec_lst = best300
                return remove_seen(seen_song, rec_lst)[:100]
    # songs_len이 0인 경우
    else: 
        rec_lst = best300
        return remove_seen(seen_song, rec_lst)[:100]

def rec_100(df):
    rec_100 = []
    no_sim = []
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        if df['songs'][i] in sim_df.columns:
            rec_100 += df['rec_list'][i]
        else:
            no_sim += df['rec_list'][i]
    return rec_100 + list(set(no_sim))[:100-len(rec_100)]

def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

###### main ######

path = '/home/nsun/mountainGo/'

# 데이터프레임 불러오기
song_mel = pd.read_csv(path+'song_mel.csv')
song_mel.head()

labels = song_mel[['song_id']]
df= song_mel.drop(columns=['song_id'])

# 유사도 구하기
similarity = cosine_similarity(df)   
sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)

# 데이터 불러오기
train = pd.read_json(path+'train.json')
val = pd.read_json(path+'val.json')
val['songs_len'] = val['songs'].apply(lambda x: len(x))

# 전역변수 생성
train_val = pd.concat([train, val])
best300 = train_val.songs.explode().value_counts()[:300].index.tolist()
train_songs = train['songs'].explode()

# 플리 내 song별로 각각 추천곡 생성
df_val = val[['id','songs','songs_len']].copy().explode('songs')
df_val['rec_list'] = df_val.progress_apply(lambda x: rec_each_song(x['id'], x['songs'], x['songs_len']), axis=1)

# 최종 100개 곡 추천
final = df_val.groupby('id').apply(lambda x: rec_100(x))
lst = []
for i in tqdm(range(len(final))):
    d = {}
    d['id'] = final.index[i]
    d['songs'] = final[final.index[i]]
    lst.append(d)

# json 파일로 저장
with open(path+'results.json', 'w', encoding='utf-8') as f:
    json.dump(lst, f, ensure_ascii=False, default=_conv) # 한글깨짐 방지

