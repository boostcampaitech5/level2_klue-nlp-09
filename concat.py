import pandas as pd

per = pd.read_csv("./prediction/per.csv")
org = pd.read_csv("./prediction/org.csv")

concat_df = pd.concat([per, org])

sort_df = concat_df.sort_values('id')
# sort_df = sort_df.drop('Unnamed: 0', axis=1)
sort_df.set_index('id', inplace=True)

#sort_df = sort_df.iloc[:, 1:]
sort_df.to_csv('./prediction/submission.csv')