##% import test datased

import pandas as pd
df = pd.read_csv('CSP_DNA_sensors.csv')

#%% data cleaning + encoding

## Check for missing values (none here)
# df.isnull().values.any()
# TODO: check how NAs are dealed with

## Look for non-number columns (sequences)
# types = df.dtypes
# for dtype in types.unique():
#     column_names = ", ".join(types[types == dtype].index)
#     print(f"Columns of type {dtype}: {column_names}")

## One-hot encoding for type_sensor and type_analyte
df = pd.get_dummies(df, columns=['type_sensor', 'type_analyte'])

#%% splitting for ATGC and no ATGC

seq_columns = ['whole_seq', 
    'local_loop_seq', 
    'anlt_seq_arms', 
    'seq_1arm',
    'seq_1arm_mod', 
    'seq_2arm', 
    'seq_2arm_mod', 
    'seq_3arm', 
    'seq_3arm_mod']

df_noseq = df.drop(seq_columns, axis = 1)

#%% Training for no ATGC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df_noseq.drop('status', axis = 1)
y = df_noseq['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#%% metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

def evaluate(y_test, y_pred):
    accuracy = int(accuracy_score(y_test, y_pred)*100)
    roc_auc = int(roc_auc_score(y_test, y_pred)*100)
    confusion = confusion_matrix(y_test, y_pred)
    precision = int(precision_score(y_test, y_pred)*100)
    recall = int(recall_score(y_test, y_pred)*100)
    f1 = int(f1_score(y_test, y_pred)*100)
    
    print(f'Accuracy: {accuracy}%')
    print(f'ROC AUC: {roc_auc}%')
    print(f'Confusion Matrix: \n{confusion}')
    print(f'Precision: {precision}%')
    print(f'Recall: {recall}%')
    print(f'F1 Score: {f1}%')

print('Evaluation for noseq all features')
evaluate(y_test, y_pred)

##feature selection
importances = model.feature_importances_
feature_importances = pd.Series(importances*100, index=X.columns).sort_values(ascending=False)

print('\nTop features for noseq:\n', feature_importances[:9])

#%% Repeat for 10 features

# X = df_noseq[feature_importances[:10].index.tolist()]
# y = df_noseq['status']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# evaluate(y_test, y_pred)

#%% K-mer DNA embedding
def count_kmers(sequence, k):
    d = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer not in d:
            d[kmer] = 1
        else:
            d[kmer] += 1
    return d

#%%
from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse=False)

for i in range(len(seq_columns)):
    kmers = df[seq_columns[i]].apply(lambda x: count_kmers(x, 2))
    kmers_dict = kmers.tolist()
    kmers_df = pd.DataFrame(vectorizer.fit_transform(kmers_dict), columns=[seq_columns[i] + '_' + x for x in vectorizer.feature_names_])
    df_other_features = df.drop(columns=seq_columns[i], axis = 1)
    df = pd.concat([df_other_features.reset_index(drop=True), kmers_df.reset_index(drop=True)], axis=1)

#%%
X = df.drop('status', axis = 1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('\nEvaluation for seq all features:')
evaluate(y_test, y_pred)

importances = model.feature_importances_
feature_importances = pd.Series(importances*100, index=X.columns).sort_values(ascending=False)

print('\nTop features for seq:\n', feature_importances[:9])
