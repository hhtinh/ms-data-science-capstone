import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot') # Look pretty

df_train_label = pd.read_csv('Training_set_labels.csv', header=0, index_col=None)

df_train_label.info()
df_train_label.head(10)

df_train_label.describe()
df_train_label.repayment_rate.median()

df_train_label.repayment_rate.plot.hist(alpha=0.75)

df_train = pd.read_csv('Training_set_values.csv', header=0, index_col=None)

df_train_label.set_index('row_id', inplace=True)
df_train.set_index('row_id', inplace=True)

df1 = pd.concat([df_train.school__ownership, df_train_label.repayment_rate], axis=1, join='inner')
df1.repayment_rate.groupby(df1.school__ownership).median()

df2 = pd.concat([df_train.admissions__sat_scores_average_overall, df_train_label.repayment_rate], axis=1, join='inner')
df2.admissions__sat_scores_average_overall.describe()
df2.corr()

df3 = pd.concat([df_train.student__demographics_median_family_income, df_train_label.repayment_rate], axis=1, join='inner')
df3.corr()

df4 = pd.concat([df_train.school__region_id, df_train_label.repayment_rate], axis=1, join='inner')
df4.repayment_rate.groupby(df4.school__region_id).median()
