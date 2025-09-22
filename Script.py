# main 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from PIL import Image
import os 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# Load dataset
df = pd.read_csv('dataset.csv')
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.drop(index=df[df['Age'] > 80].index.tolist(), inplace=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
cat_cols = ['Geography', 'Gender']
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(cat_cols))

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
ready_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

all_pipeline = ColumnTransformer([
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols),
    ('ready_pipeline', ready_pipeline, ready_cols)
])

X_train_prepared = all_pipeline.fit_transform(X_train)
X_test_prepared = all_pipeline.transform(X_test)

# Class weights
no_bin_class = 1 - np.bincount(y_train) / X_train.shape[0]
class_weight = {0: no_bin_class[0], 1: no_bin_class[1]}

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prepared, y_train)

# Init metrics file
with open('metrics.txt','w') as f:
    pass

def train_model(xtrain, ytrain, plot_name='', class_weight=None):
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=15, random_state=42, 
        criterion='gini', class_weight=class_weight
    )
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(X_test_prepared)

    score_test = f1_score(y_test, ypred)
    score_train = f1_score(ytrain, clf.predict(xtrain))
    clf_name = clf.__class__.__name__

    # plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, ypred), annot=True, fmt='d', cmap='Blues')
    plt.title(plot_name)
    plt.xticks(ticks=np.arange(2)+0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2)+0.5, labels=[False, True])
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    with open('metrics.txt','a') as f:
        f.write(f'Model name: {clf_name}\n')
        f.write(f'Train f1_score: {score_train}\n')
        f.write(f'Test f1_score: {score_test}\n')
        f.write('----------'*2 + '\n')

# Run models
train_model(X_train_prepared, y_train, plot_name='Baseline')
train_model(X_train_prepared, y_train, plot_name='Class-Weight', class_weight=class_weight)
train_model(X_train_resampled, y_train_resampled, plot_name='SMOTE')

# Combine confusion matrices
confusion_matrix_paths = ['Baseline.png', 'Class-Weight.png', 'SMOTE.png']
plt.figure(figsize=(15, 5))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')
plt.suptitle("RandomForest", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=300)

# Cleanup
for path in confusion_matrix_paths:
    os.remove(path)
