import pandas as pd
import numpy as np
from tqdm import tqdm
from seqfold import dg

df = pd.read_csv('dataset.csv')

# dropping ref columns columns
df = df.drop(['results_ref', 'sensor_ref'], axis=1)

# one-hot encoding for sensor, core and analyte types
type_columns = df.filter(like='type_')
for col in type_columns:
    one_hot_encoded = pd.get_dummies(df[col], prefix=col.replace('type_', ''))
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop(col, axis=1)
    
# dropipnf darta that depends on sensor, analyte and core types:
df = df.drop(['incube_time_H', 'Na_mM', 'Mg_mM', 'analyte_nM', 'sensor_nM', 'f_sub_nM', '1arm_nM', '2arm_nM', '3arm_nM', '4arm_nM'], axis=1)

# calculating gibbs energy (with tqdm so that I won't go bonky)
seq_cols = df.select_dtypes(include='object').columns
total_operations = len(seq_cols) * len(df)
tqdm.pandas()

for col in tqdm(seq_cols, desc="Processing columns"):
    df[col] = df[col].str.replace('С', 'C').str.replace('А', 'A').str.replace('Т', 'T')
    df[col] = df.progress_apply(
        lambda row: dg(row[col], row['temp_C']) if not pd.isna(row[col]) else row[col], 
        axis=1
    )
    
df['success'] = df['ctrl_pos'] / df['ctrl_neg']
df['class_2'] = np.where(df['success'] > 2, 1, 0)
df = df.drop(['success', 'ctrl_pos', 'ctrl_neg'], axis = 1)
df.to_csv('dataset_dg.csv', index=False)
