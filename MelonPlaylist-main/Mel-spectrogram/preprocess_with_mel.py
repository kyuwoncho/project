from operator import itemgetter
from scipy.sparse import coo_matrix, save_npz
import numpy as np
from collections import defaultdict

import os
import io
import distutils.dir_util
from collections import Counter
import json
import pickle 

def pickle_dump(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./" + parent)
    with io.open("./" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)
    return json_obj

def most_popular(playlists, col, topk_count):
    '''가장 많이 등장한 곡/태그를 topk_count만큼 추출하는 함수'''
    c = Counter()
    for doc in playlists:
        c.update(doc[col])
    topk = c.most_common(topk_count)
    return c, [k for k, v in topk]

def remove_seen(seen, l):
    '''l 중에서 seen에 이미 있는 아이템은 제거하는 함수'''
    seen = set(seen)
    return [x for x in l if not (x in seen)]


####### main #######


if not(os.path.isdir('data/')):
    os.makedirs('data/')

train = load_json('/data/mountain/train.json')
val = load_json('/data/mountain/val.json')
test = load_json('/data/mountain/test.json')
train = train + val + test

print('rank popular songs/tags...')
data_by_yearmonth = defaultdict(list)
# data_by_yearmonth에 연도별, 년월별로 플레이리스트를 추가  
# data_by_yearmonth = {2013: [플레이리스트1, 플레이리스트2, ...], 2013-07: [플레이리스트1, 플레이리스트2, ...]}
for q in train:
    try:
        data_by_yearmonth[q['updt_date'][0:4]].append(q)
    except:
        pass
    try:
        data_by_yearmonth[q['updt_date'][0:7]].append(q)
    except:
        pass
data_by_yearmonth = dict(data_by_yearmonth)

# 가장 많이 등장한 곡 200개를 뽑아서 most_popular_results에 저장
# most_popular_results = {'songs': [곡1, 곡2, ...]}
most_popular_results = {}
songs_mp_counter, most_popular_results['songs'] = most_popular(train, "songs", 200)

# 연도별, 년월별로 가장 많이 등장한 곡 200개 most_popular_results에 저장 (key)
# most_popular_results = {'songs': [곡1, 곡2, ...], 'tags': [태그1, 태그2, ...], 
#                         'songs2013': [곡1, 곡2, ...], 'tags2013': [태그1, 태그2, ...], 
#                         'songs2013-07': [곡1, 곡2, ...], 'tags2013-07': [태그1, 태그2, ...], ...}
for y in data_by_yearmonth.keys():
    _, most_popular_results['songs' + y] = most_popular(data_by_yearmonth[y], "songs", 200)
   
   
print('write train matrix...')
playlist_song_train_matrix = []
p_encode, s_encode, p_decode, s_decode = {}, {}, {}, {}
playlist_idx = 0
song_idx = 0

# p_encode : {플리 id : 플리idx, ...}
# s_encode : {곡 id : 곡idx, ...}
# playlist_song_train_matrix : [[플리idx, 곡idx], ...]
for q in train:
    if len(q['songs']) >= 1:
        p_encode[q['id']] = playlist_idx
        for s in q['songs']:
            if s not in s_encode.keys():
                s_encode[s] = song_idx
                song_idx += 1
            playlist_song_train_matrix.append([playlist_idx, s_encode[s]])
        playlist_idx += 1

# coo_matrix : sparse matrix (data, (row, col)) 형태로 저장 (shape: 행렬의 크기)
# playlist_song_train_matrix : [[플리idx, 곡idx], ...]
playlist_song_train_matrix = np.array(playlist_song_train_matrix)
playlist_song_train_matrix = coo_matrix((np.ones(playlist_song_train_matrix.shape[0]),  # data 
                                         (playlist_song_train_matrix[:,0],              # row : 플리idx
                                          playlist_song_train_matrix[:,1])),            # col : 곡/태그/제목태그idx
                                        shape=(playlist_idx,song_idx))                  # shape : (플리id 개수, 곡/태그/제목태그 개수)
# playlist_song_train_matrix 원소 : 1 (플리id에 곡/태그/제목태그가 포함되어 있으면 1, 아니면 0)

# [플리idx, 곡/태그/제목태그idx] sparse matrix를 npz 파일로 저장
save_npz('data/playlist_song_train_matrix_with_mel.npz', playlist_song_train_matrix)

# s_decode : {곡 idx : 곡 id, ...}
for s in s_encode.keys():
    s_decode[s_encode[s]] = s
# p_encode, s_decode를 pickle 파일로 저장
# p_encode : {플리 id : 플리idx, ...}
pickle_dump(s_decode, 'data/song_label_decoder.pickle')
pickle_dump(p_encode, 'data/playlist_label_encoder.pickle')
    
print('write val item indices...')
for q in val:
    # songs에 하나라도 값이 있는 경우 
    if len(q['songs']) >= 1:
        # (테스트셋 곡과 가장 많이 등장한 200개 곡의 교집합)에 해당하는 곡들의 빈도수 평균이 0을 초과하는 경우
        if np.mean([songs_mp_counter[i] for i in q['songs']]) > 0:
            items = [s_encode[s] for s in q['songs']]  # items : [곡idx, 곡idx, ...]
            # test 데이터셋에 items 키 추가
            q['items'] = items
    
    # 테스트셋 플리 년월에 해당하는 'songsYYYY-MM'이 most_popular_results에 있는 경우
    if 'songs'+q['updt_date'][0:7] in most_popular_results.keys():
        # 테스트셋에 'songs_mp' 키 추가 <- {(songsYYYY-MM에 있는 곡 | 가장 인기있는 곡) - 테스트셋 플리에 있는 곡} 100개
        q['songs_mp'] = (remove_seen(q['songs'], most_popular_results['songs'+q['updt_date'][0:7]] + remove_seen(most_popular_results['songs'+q['updt_date'][0:7]], most_popular_results['songs'])))[:100]
    
    # 테스트셋 플리 연도에 해당하는 'songsYYYY'이 most_popular_results에 있는 경우
    elif 'songs'+q['updt_date'][0:4] in most_popular_results.keys():
        # 테스트셋에 'songs_mp' 키 추가 <- {(songsYYYY에 있는 곡 | 가장 인기있는 곡) - 테스트셋 플리에 있는 곡} 100개
        q['songs_mp'] = (remove_seen(q['songs'],most_popular_results['songs'+q['updt_date'][0:4]] + remove_seen(most_popular_results['songs'+q['updt_date'][0:4]], most_popular_results['songs'])))[:100]
    
    # 테스트셋 플리 연도 및 연월에 해당하는 'songsYYYY-MM'과 'songsYYYY 둘 다 most_popular_results에 없는 경우
    else:
        # 테스트셋에 'songs_mp' 키 추가 <- {가장 인기있는 곡 - 테스트셋 플리에 있는 곡} 100개
        q['songs_mp'] = remove_seen(q['songs'], most_popular_results['songs'][:100])


# song별 mel-spectrogram의 평균값(scalar) 
with open('/data/mountain/mel_mean.pkl', 'rb') as f:
    mel_mean = pickle.load(f)
# mel_mean의 song_id 키를 index로 변환
for id in list(mel_mean.keys()):
    if id in s_encode.keys():
        idx = s_encode[id]
        mel_mean[idx]=mel_mean.pop(id)
    else:
        mel_mean.pop(id)

# 'items', 'songs_mp' 키를 추가한 val 데이터셋을 json 파일로 저장
write_json(val, 'data/val_items.json')
with open('/home/nsun/mountain/data/mel_mean_idx.pkl', 'wb') as f:
    pickle.dump(mel_mean, f)
print('End preprocessing')
