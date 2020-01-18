import pandas as pd
import numpy as np

data =  pd.read_csv('parkinsons_updrs.data')

del data['subject#']
del data['sex']
data.drop(data.columns[4], axis=1,inplace=True)
data.drop(data.columns[8], axis=1,inplace=True)
data.drop(data.columns[7], axis=1,inplace=True)
data.drop(data.columns[12], axis=1,inplace=True)

np.where(np.tril(data.corr().abs().values,-1) > 0.98)
data.describe().T

data.to_csv('parkinsons_updrs_processed.data',index=False)
