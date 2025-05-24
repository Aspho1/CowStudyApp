import pandas as pd
import numpy as np
# pred_path = '/opt/CowStudyApp/app_data/analysis/predictions/website_admin/RB_22/HMM/predictions.csv'
pred_path = '/opt/CowStudyApp/app_data/analysis/predictions/website_admin/RB_22/LSTM/opo/predictions.csv'

df = pd.read_csv(pred_path)

print(df.shape)


print(sum(pd.isna(df.predicted_state)))
dff = df[~pd.isna(df.activity)]

print(dff.head())
print(dff.shape)

print(dff['predicted_state'].describe())

print(np.average(dff['activity'] == dff['predicted_state']))

