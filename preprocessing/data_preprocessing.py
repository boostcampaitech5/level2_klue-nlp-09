import pandas as pd
import re
def clean_foreign_language(df):
  def clean_chinese(row):
    # print("row: ",row)
    foreign_regex = re.compile(r'([一-鿕]|[㐀-䶵]|[豈-龎])+')
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


if  __name__ == "__main__":
  df = pd.read_csv("../dataset/train/v1/train.csv")
  foreign_regex = re.compile(r'([一-鿕]|[㐀-䶵]|[豈-龎])+') 
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


  