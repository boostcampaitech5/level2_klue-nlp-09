import pandas as pd
import random
random.seed(42)
import pickle
from konlpy.tag import *
from collections import defaultdict

def spellcheck(df):
  return df


def swap_text(df):
  new_df = df
  return new_df

def synonym_replacement(data, n=2):
    wordnet = {}
    with open("wordnet.pickle", "rb") as f:
        wordnet = pickle.load(f)

    # wordnet 유사어 집합 내 타겟과 동일한 단어 제거
    useless_keys = []
    for key, values in wordnet.items():
        if wordnet[key] == [key]:
            useless_keys.append(key)
        if key in values:
            values.remove(key)

    for uk in useless_keys:
        del wordnet[uk]

    # synonym replacement 메인
    replaced_dict = defaultdict(list)
    for idx, d in data.iterrows():
        sentence = d.sentence
        replaced_sentence = sentence
        sbj_dict, obj_dict = eval(d.subject_entity), eval(d.object_entity)
        
        # 형태소 단위로 분리하여 유의어 탐색
        okt = Okt()
        words = [ss for ss in okt.morphs(sentence) if len(ss) > 1] # 1글자 유의어는 틀릴 확률이 너무 높기 때문에, 2글자 이상 단어만
        random_words = list(set([word for word in words]))
        random.shuffle(random_words)
        
        # 형태소 단위 단어 중 최대 n개의 단어를 유의어로 교체
        num_replaced= 0
        replaced_indices = []
        original_words, replaced_words = [], []
        len_changes = []
        for random_word in random_words:
            if random_word not in sbj_dict['word'] and random_word not in obj_dict['word']: # entity에 들어있지 않은 단어들만
                synonyms = get_synonyms(random_word)
                if synonyms:
                    replaced_index = replaced_sentence.find(random_word)
                    replaced_indices.append(replaced_index)
                    
                    synonym = random.choice(list(synonyms))
                    original_words.append(random_word)
                    replaced_words.append(synonym)
                    len_changes.append(len(synonym) - len(random_word))
                    replaced_sentence = replaced_sentence.replace(random_word, synonym, 1) # 중복 단어인 경우 1개만 변경. 몇 번째 단어를 바꿀지도 정할 수 있으면 좋음(나중에).
                    num_replaced += 1
                    
            if num_replaced == n:
                break
        
        # 교체된 단어의 길이가 다른 경우, subject_entity와 object_entity의 index가 바뀌었으면 변경하여 저장
        for replaced_index, len_change in zip(replaced_indices, len_changes):
            
            if replaced_index < sbj_dict['start_idx']:
                if replaced_index < obj_dict['start_idx']:
                    sbj_dict['start_idx'] = sbj_dict['start_idx'] + len_change
                    sbj_dict['end_idx'] = sbj_dict['end_idx'] + len_change
                    obj_dict['start_idx'] = obj_dict['start_idx'] + len_change
                    obj_dict['end_idx'] = obj_dict['end_idx'] + len_change
                else:
                    sbj_dict['start_idx'] = sbj_dict['start_idx'] + len_change
                    sbj_dict['end_idx'] = sbj_dict['end_idx'] + len_change
            else:
                if replaced_index < obj_dict['start_idx']:
                    obj_dict['start_idx'] = obj_dict['start_idx'] + len_change
                    obj_dict['end_idx'] = obj_dict['end_idx'] + len_change
        
        # 증강된 데이터 저장을 위해 dict 형태로 제작
        if sentence != replaced_sentence:
            replaced_dict['sentence'].append(replaced_sentence)
            replaced_dict['id'].append(100000 + d.id) # 증강되어 추가된 데이터의 id는 기존 id + 100000
            replaced_dict['subject_entity'].append(str(sbj_dict))
            replaced_dict['object_entity'].append(str(obj_dict))
            replaced_dict['label'].append(d.label)
            replaced_dict['source'].append(d.source)
            replaced_dict['original'].append(original_words)
            replaced_dict['replaced'].append(replaced_words)
            
        # if idx == 9: # early stop for test
        #     break
        
        if (idx + 1) % 1000 == 0:
            print(f'{idx + 1} data are preprocessed.')
            
    print('All data are preprocessed.')
            
    replaced_data = pd.DataFrame.from_dict(replaced_dict)
    augmented_data = pd.concat([data, replaced_data], axis=0, ignore_index=True)

    def get_synonyms(word):
      synonyms = []
      try:
          for syn in wordnet[word]:
              synonyms.append(syn)
      except:
          pass

      return synonyms

    return augmented_data