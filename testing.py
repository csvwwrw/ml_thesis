import pandas as pd

df = pd.read_csv('CSP_DNA_sensors.csv')

df = pd.get_dummies(df, columns=['type_sensor', 'type_analyte'])

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# assuming X is your feature set and y is the target variable
X = df.drop(list(df.filter(regex='_seq')), axis = 1)
X = X.drop(list(df.filter(regex='seq_\d+arm')), axis = 1)
X = X.drop('status', axis = 1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# feature selection
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print(feature_importances)

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

sequence = 'AGTCAG'
k = 2
print(count_kmers(sequence, k))

#%%
df['whole_seq'] = df['whole_seq'].apply(lambda x: count_kmers(x, 2))

#%%
from sklearn.feature_extraction import DictVectorizer



X_dict = df['whole_seq'].tolist()
vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(X_dict)

#%%

from sklearn.ensemble import RandomForestClassifier

# Assuming 'kmers' is your feature and 'label' is what you're trying to predict

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the model
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Now you can use the model to make predictions
predictions = model.predict(X_test)

#%%
from sklearn.metrics import f1_score
# Assume y_true is your true labels
f1 = f1_score(y_test, predictions, average='macro')
print(f1)