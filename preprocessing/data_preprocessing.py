import pandas as pd
import random
random.seed(42)
import pickle
from konlpy.tag import *
from collections import defaultdict
import re


def clean_foreign_language(df):
    def clean_chinese(row):
        # foreign_regex = re.compile(r'(([一-鿕]|[㐀-䶵]|[豈-龎])+)()')
        foreign_regex = re.compile(r'(\({1}(([一-鿕]|[㐀-䶵]|[豈-龎])+)\){1})|((([一-鿕]|[㐀-䶵]|[豈-龎])+)\,{1}\s{1})')
        sentence = row['sentence']
        while(re.search(foreign_regex, row['sentence'])):
            # sentence 의 한자 제거
            start=re.search(foreign_regex, row['sentence']).span()[0]
            end = re.search(foreign_regex, row['sentence']).span()[1]
            row['sentence'] = row['sentence'][0:start] + row['sentence'][end:len(row['sentence'])]
            # 한자 제거에 따른 subject, object entity index 변경
            temp_sub = row['subject_entity'].split("'")
            temp_obj = row['object_entity'].split("'")
            sub_start = int(temp_sub[6].split(" ")[1].split(",")[0])
            sub_end = int(temp_sub[8].split(" ")[1].split(",")[0])
            obj_start = int(temp_obj[6].split(" ")[1].split(",")[0])
            obj_end = int(temp_obj[8].split(" ")[1].split(",")[0])
            length = end - start
            if sub_start >start:
                sub_start = sub_start-length
                sub_end = sub_end-length
                temp_sub[6] = ": "+str(sub_start) + ", "
                temp_sub[8] = ": "+str(sub_end) + ", "
                row['subject_entity'] = "'".join(temp_sub)
            if obj_start >start:
                obj_start = obj_start-length
                obj_end = obj_end-length
                temp_obj[6] = ": "+str(obj_start) + ", "
                temp_obj[8] = ": "+str(obj_end) + ", "
                row['object_entity'] = "'".join(temp_obj)

    def clean_chinese(row):
        # print("row: ",row)
        foreign_regex = re.compile(r'([一-鿕]|[㐀-䶵]|[豈-龎])+')
        sentence = row['sentence']
        # print(sentence)
        while(re.search(foreign_regex, row['sentence'])):
            # sentence 의 한자 제거
            start=re.search(foreign_regex, row['sentence']).span()[0]
            end = re.search(foreign_regex, row['sentence']).span()[1]
            row['sentence'] = row['sentence'][0:start] + row['sentence'][end:len(row['sentence'])]
            # 한자 제거에 따른 subject, object entity index 변경
            temp_sub = row['subject_entity'].split("'")
            temp_obj = row['object_entity'].split("'")
            sub_start = int(temp_sub[6].split(" ")[1].split(",")[0])
            sub_end = int(temp_sub[8].split(" ")[1].split(",")[0])
            obj_start = int(temp_obj[6].split(" ")[1].split(",")[0])
            obj_end = int(temp_obj[8].split(" ")[1].split(",")[0])
            length = end - start
            if sub_start >start:
                sub_start = sub_start-length
                sub_end = sub_end-length
                temp_sub[6] = ": "+str(sub_start) + ", "
                temp_sub[8] = ": "+str(sub_end) + ", "
                row['subject_entity'] = "'".join(temp_sub)
            if obj_start >start:
                obj_start = obj_start-length
                obj_end = obj_end-length
                temp_obj[6] = ": "+str(obj_start) + ", "
                temp_obj[8] = ": "+str(obj_end) + ", "
                row['object_entity'] = "'".join(temp_obj)

    return row

    search_subject = []
    search_object = []
    for i, dic in enumerate(df['subject_entity']):
        # print(dic)
        if re.search(foreign_regex,dic):
            search_subject.append(i)
    for i, dic in enumerate(df['object_entity']):
        # print(dic)
        if re.search(foreign_regex,dic):
            search_object.append(i)

    df.drop(search_subject, axis =0, inplace=True)
    df.drop(search_object, axis =0, inplace=True)
    df = df.apply(lambda x : clean_chinese(x) , axis=1)
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
        
    def get_synonyms(word):
        synonyms = []
        try:
            for syn in wordnet[word]:
                synonyms.append(syn)
        except:
            pass

        return synonyms

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
        for random_word in random_words:
            if random_word in sbj_dict['word'] or sbj_dict['word'] in random_word or random_word in obj_dict['word'] or obj_dict['word'] in random_word:
                random_words.remove(random_word)
        random.shuffle(random_words)
        
        # 형태소 단위 단어 중 최대 n개의 단어를 유의어로 교체
        num_replaced= 0
        replaced_indices = []
        original_words, replaced_words = [], []
        len_changes = []
        for random_word in random_words:
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
        
#         if (idx + 1) % 1000 == 0:
#             print(f'{idx + 1} data are preprocessed.')
            
#     print('All data are preprocessed.')
            
    replaced_data = pd.DataFrame.from_dict(replaced_dict)
    augmented_data = pd.concat([data, replaced_data], axis=0, ignore_index=True)

    return augmented_data


def random_deletion(data, n=1):
    replaced_dict = defaultdict(list)
    for idx, d in data.iterrows():
        sentence = d.sentence
        replaced_sentence = sentence
        sbj_dict, obj_dict = eval(d.subject_entity), eval(d.object_entity)
        
        # 형태소 단위로 분리
        okt = Okt()
        words = [ss for ss in okt.morphs(sentence) if len(ss) > 1] # 2글자 이상 단어만
        random_words = list(set([word for word in words]))
        for random_word in random_words:
            if random_word in sbj_dict['word'] or sbj_dict['word'] in random_word or random_word in obj_dict['word'] or obj_dict['word'] in random_word:
                random_words.remove(random_word)
        random.shuffle(random_words)
        
        # 단어 랜덤 삭제
        num_replaced = 0
        replaced_indices = []
        original_words, replaced_words = [], []
        len_changes = []
        for random_word in random_words:
            match = re.search(r"[(){}\[\]<>]", random_word)
            if match:
                continue
            replaced_index = replaced_sentence.find(random_word)
            replaced_indices.append(replaced_index)

            original_words.append(random_word)
            replaced_words.append('')
            replaced_sentence = replaced_sentence.replace(random_word, '', 1) # 중복 단어인 경우 1개만 삭제
            if '  ' in replaced_sentence:
                replaced_sentence = replaced_sentence.replace('  ', ' ') # 단어 삭제로 인한 이중 띄어쓰기 삭제
                len_changes.append(-1 * (len(random_word) + 1))
            else:
                len_changes.append(-1 * len(random_word))
            num_replaced += 1
            
            if num_replaced == n:
                break
            
        # subject_entity와 object_entity의 index가 바뀌었으면 변경하여 저장
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
            replaced_dict['id'].append(200000 + d.id) # 증강되어 추가된 데이터의 id는 기존 id + 200000 (RD)
            replaced_dict['subject_entity'].append(str(sbj_dict))
            replaced_dict['object_entity'].append(str(obj_dict))
            replaced_dict['label'].append(d.label)
            replaced_dict['source'].append(d.source)
            replaced_dict['original'].append(original_words)
            replaced_dict['replaced'].append(replaced_words)

        # if idx == 9: # early stop for test
        #     break

        # if (idx + 1) % 1000 == 0:
        #     print(f'{idx + 1} data are preprocessed.')
            
    # print('All data are preprocessed.')
            
    replaced_data = pd.DataFrame.from_dict(replaced_dict)
    augmented_data = pd.concat([data, replaced_data], axis=0, ignore_index=True)

    return augmented_data


def random_swap(data, n_pairs=1):
    replaced_dict = defaultdict(list)
    for idx, d in data.iterrows():
        sentence = d.sentence
        replaced_sentence = sentence
        sbj_dict, obj_dict = eval(d.subject_entity), eval(d.object_entity)
        
        # 형태소 단위로 분리
        okt = Okt()
        words = [ss for ss in okt.nouns(sentence) if len(ss) > 1] # 2글자 이상 '명사' 단어만
        random_words = list(set([word for word in words]))
        for random_word in random_words:
            if random_word in sbj_dict['word'] or sbj_dict['word'] in random_word or random_word in obj_dict['word'] or obj_dict['word'] in random_word:
                random_words.remove(random_word)
        random.shuffle(random_words)
        
        # 단어 랜덤 스왑
        num_replaced = 0
        replaced_indices = []
        original_words, replaced_words = [], []
        len_changes = []
        random_words_pairs = [[random_word1, random_word2] for random_word1, random_word2 in zip(random_words[:-1], random_words[1:])]
        for random_words_pair in random_words_pairs:
            random_word1, random_word2 = random_words_pair[0], random_words_pair[1]
            replaced_index1, replaced_index2 = replaced_sentence.find(random_word1), replaced_sentence.find(random_word2)
            if replaced_index1 > replaced_index2:
                random_word1, random_word2 = random_word2, random_word1
                replaced_index1, replaced_index2 = replaced_index2, replaced_index1
            replaced_indices.extend([replaced_index1, replaced_index2])

            original_words.extend([random_word1, random_word2])
            replaced_words.extend([random_word2, random_word1])
            len_changes.extend([len(random_word2) - len(random_word1), len(random_word1) - len(random_word2)])
            replaced_sentence1 = replaced_sentence.replace(random_word1, random_word2, 1)
            replaced_sentence2 = replaced_sentence.replace(random_word2, random_word1, 1)
            replaced_sentence = replaced_sentence1[:replaced_index1 + len(random_word2) + 1] + replaced_sentence2[replaced_index1 + len(random_word2) + 1:]

            num_replaced += 1
            
            if num_replaced == n_pairs:
                break
            
        # subject_entity와 object_entity의 index가 바뀌었으면 변경하여 저장
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
            replaced_dict['id'].append(300000 + d.id) # 증강되어 추가된 데이터의 id는 기존 id + 300000 (RS)
            replaced_dict['subject_entity'].append(str(sbj_dict))
            replaced_dict['object_entity'].append(str(obj_dict))
            replaced_dict['label'].append(d.label)
            replaced_dict['source'].append(d.source)
            replaced_dict['original'].append(original_words)
            replaced_dict['replaced'].append(replaced_words)

        # if idx == 9: # early stop for test
        #     break

#         if (idx + 1) % 1000 == 0:
#             print(f'{idx + 1} data are preprocessed.')
            
#     print('All data are preprocessed.')
            
    replaced_data = pd.DataFrame.from_dict(replaced_dict)
    augmented_data = pd.concat([data, replaced_data], axis=0, ignore_index=True)

    return augmented_data


def random_insertion(data, n=1):
    wordnet = {}
    with open("wordnet.pickle", "rb") as f:
        wordnet = pickle.load(f)
        
    replaced_dict = defaultdict(list)
    for idx, d in data.iterrows():
        sentence = d.sentence
        replaced_sentence = sentence
        sbj_dict, obj_dict = eval(d.subject_entity), eval(d.object_entity)
        
        # 형태소 단위로 분리
        okt = Okt()
        words = [ss for ss in okt.morphs(sentence) if len(ss) > 1] # 2글자 이상 단어만
        random_words = list(set([word for word in words]))
        for random_word in random_words:
            if random_word in sbj_dict['word'] or sbj_dict['word'] in random_word or random_word in obj_dict['word'] or obj_dict['word'] in random_word:
                random_words.remove(random_word)
        random.shuffle(random_words)
        
        # 단어 랜덤 삽입
        num_replaced = 0
        replaced_indices = []
        original_words, replaced_words = [], []
        len_changes = []
        for random_word in random_words:
            random1 = random.randint(0, len(wordnet) - 1)
            random2 = random.randint(0, len(list(wordnet.values())[random1]) - 1)
            inserted_word = list(wordnet.values())[random1][random2]
            
            replaced_index = replaced_sentence.find(random_word)
            replaced_indices.append(replaced_index)

            original_words.append('')
            replaced_words.append(inserted_word)
            len_changes.append(len(inserted_word) + 1)
            replaced_sentence = replaced_sentence[:replaced_index] + inserted_word + ' ' + replaced_sentence[replaced_index:]
            
            num_replaced += 1
            
            if num_replaced == n:
                break
            
        # subject_entity와 object_entity의 index가 바뀌었으면 변경하여 저장
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
            replaced_dict['id'].append(400000 + d.id) # 증강되어 추가된 데이터의 id는 기존 id + 400000 (RI)
            replaced_dict['subject_entity'].append(str(sbj_dict))
            replaced_dict['object_entity'].append(str(obj_dict))
            replaced_dict['label'].append(d.label)
            replaced_dict['source'].append(d.source)
            replaced_dict['original'].append(original_words)
            replaced_dict['replaced'].append(replaced_words)

        # if idx == 9: # early stop for test
        #     break

#         if (idx + 1) % 1000 == 0:
#             print(f'{idx + 1} data are preprocessed.')
            
#     print('All data are preprocessed.')
            
    replaced_data = pd.DataFrame.from_dict(replaced_dict)
    augmented_data = pd.concat([data, replaced_data], axis=0, ignore_index=True)

    return augmented_data


def entity_replacement(data, threshold=1000, entity='subject'):
                    
    def ent2df(data):
        sbj_dict = data.subject_entity
        obj_dict = data.object_entity
        sbj_dd, obj_dd = defaultdict(list), defaultdict(list)
        for sd, od in zip(sbj_dict, obj_dict):
            sd, od = eval(sd), eval(od)
            for (sd_key, sd_value), (od_key, od_value) in zip(sd.items(), od.items()):
                sbj_dd[sd_key].append(sd_value)
                obj_dd[od_key].append(od_value)
                
        sbj_df, obj_df = pd.DataFrame.from_dict(sbj_dd), pd.DataFrame.from_dict(obj_dd)
        
        return sbj_df, obj_df

    def change_entity(row_data, entity='subject'):
        ent_dict = eval(row_data[entity + '_entity'])
        sentence = row_data.sentence
        original_ent = ent_dict['word']
        replaced_ent = original_ent
        tgt_df = sbj_df if entity == 'subject' else obj_df
        while replaced_ent == original_ent:
            tgt_df_same_type = tgt_df[tgt_df.type == ent_dict['type']]
            replaced_ent = random.sample(list(tgt_df_same_type.word), 1)[0]
        
        replaced_sentence = sentence.replace(original_ent, replaced_ent)
        replaced_ent_dict = ent_dict.copy()
        replaced_ent_dict['word']  = replaced_ent
        replaced_ent_dict['start_idx'] = ent_dict['start_idx'] + (len(replaced_ent) - len(original_ent)) * sentence[:ent_dict['start_idx']].count(original_ent)
        replaced_ent_dict['end_idx'] = ent_dict['end_idx'] + (len(replaced_ent) - len(original_ent)) * (sentence[:ent_dict['start_idx']].count(original_ent) + 1)
        
        other_entity = 'object' if entity == 'subject' else 'subject'
        other_ent_dict = eval(row_data[other_entity + '_entity'])
        replaced_other_ent_dict = other_ent_dict.copy()
        if other_ent_dict['start_idx'] > ent_dict['start_idx']:
            replaced_other_ent_dict['start_idx'] = other_ent_dict['start_idx'] + (len(replaced_ent) - len(original_ent)) * sentence[:other_ent_dict['start_idx']].count(original_ent)
            replaced_other_ent_dict['end_idx'] = other_ent_dict['end_idx'] + (len(replaced_ent) - len(original_ent)) * sentence[:other_ent_dict['start_idx']].count(original_ent)
            
        replaced_row_data = {}
        replaced_row_data['id'] = 500000 + row_data.id
        replaced_row_data['sentence'] = replaced_sentence
        replaced_row_data[entity + '_entity'] = str(replaced_ent_dict)
        replaced_row_data[other_entity + '_entity'] = str(replaced_other_ent_dict)
        replaced_row_data['label'] = row_data.label
        replaced_row_data['source'] = row_data.source
        replaced_row_data['original'] = original_ent
        replaced_row_data['replaced'] = replaced_ent
        replaced_row_data = pd.DataFrame.from_dict(replaced_row_data, orient='index').T
        
        return replaced_row_data
    
    sbj_df, obj_df = ent2df(data)
    augmented_data = pd.DataFrame()
    data_label_count = data.label.value_counts()
    for key in dict(data_label_count).keys():
        label_data = data[data.label == key]
        label_data = label_data.sample(frac=1.0).reset_index(drop=True)
        if len(label_data) < threshold:
            for idx, ld in label_data.iterrows():
                replaced_ld = change_entity(ld, entity='subject')
                label_data = pd.concat([label_data, replaced_ld], ignore_index=True)

                if len(label_data) == threshold:
                    break
                    
                # if idx == 9:
                #     break

        augmented_data = pd.concat([augmented_data, label_data], ignore_index=True)
        
        print(f'[{key}] {data_label_count[key]} -> {augmented_data.label.value_counts()[key]}')
    
    print('All data are preprocessed.')
    
    return augmented_data


if  __name__ == "__main__":
    df = pd.read_csv("../dataset/train/train_90.csv")
    foreign_regex = re.compile(r'([一-鿕]|[㐀-䶵]|[豈-龎])+') 
    search_subject = []
    search_object = []
    for i, dic in enumerate(df['subject_entity']):
        # print(dic)
        if re.search(foreign_regex,dic):
            search_subject.append(i)
    for i, dic in enumerate(df['object_entity']):
        # print(dic)
        if re.search(foreign_regex,dic):
            search_object.append(i)
    print(len(df)- len(search_object)-len(search_subject))
    df.drop(search_subject, axis =0, inplace=True)
    df.drop(search_object, axis =0, inplace=True)
    print(len(df))
    # index_list = []
    # for i, sen in enumerate(df['sentence']):
    #   if re.search(foreign_regex, sen):
    #     index=df.loc[df['sentence'] == sen]
    #     # index_list.append(index)
    #     # print(index)
    #     # break

    print("before : ",df.loc[11253]['sentence'])
    df=clean_foreign_language(df)
    print("after : ", df.loc[11253]['sentence'])


    # # print(index_list)
