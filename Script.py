
# main 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from PIL import Image
import os 
# sklearn ----preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn_features.transformers import DataFrameSelector
#sklearn ----model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#sklearn ----metrics
from sklearn.metrics import f1_score,confusion_matrix
# 1- load data and preprocessing
      


df=pd.read_csv(r'dataset.csv') 
df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)
# Flitering using age feature using threshold value 80 
df.drop(index=df[df['Age']>80].index.tolist(),inplace=True)
#To feature anf target
X=df.drop('Exited',axis=1)
y=df['Exited']
# split to train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#Slice the lists of numerical and categorical features
num_cols=['CreditScore','Age','Balance','EstimatedSalary']
cat_cols=['Geography','Gender']
ready_cols=list(set(X_train.columns.tolist())-set(num_cols)-set(cat_cols))
num_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),   
    ('scaler',StandardScaler()) 
])
cat_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
ready_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')) ])
all_pipeline=ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,num_cols),  
    ('cat_pipeline',cat_pipeline,cat_cols),
    ('ready_pipeline',ready_pipeline,ready_cols)
])
X_train_prepared=all_pipeline.fit_transform(X_train)
X_test_prepared=all_pipeline.transform(X_test)
## 2-prepare class_weight for solving inbalance data
no_bin_class=1-np.bincount(y_train)/X_train.shape[0]
class_weight={0:no_bin_class[0],1:no_bin_class[1]}

# 3- use smote to solve imbalance data
smote=SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prepared, y_train)
with open('metrix.txt','w') as f:
     pass
def train_model(xtrain,ytrain,plot_name='',class_weight=None):
    global clf_name
    clf=RandomForestClassifier(n_estimators=300,max_depth=15,random_state=42,criterion='gini',class_weight=class_weight)
    clf.fit(xtrain,ytrain)
    ypred=clf.predict(X_test_prepared)
    score_test=f1_score(y_test,ypred)
    ypred_train=clf.predict(xtrain)
    score_train=f1_score(ytrain,ypred_train)
    clf_name=clf.__class__.__name__
    #plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test,ypred),annot=True,fmt='d',cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2)+0.5,labels=[False,True])
    plt.yticks(ticks=np.arange(2)+0.5,labels=[False,True])
    ## save plot
    plt.savefig(f'{plot_name}.png',bbox_inches='tight',dpi=300)
    plt.close()
    with open('metrix.txt','a') as f:
        f.write(f'Model name: {clf_name}\n')
        f.write(f'Train f1_score: {score_train}\n')
        f.write(f'Test f1_score: {score_test}\n')
        f.write('----------'*2+'\n')
        
train_model(X_train_prepared,y_train,plot_name='Without handling imbalance data')
train_model(X_train_prepared,y_train,plot_name='With-class-weight',class_weight=class_weight)
## 3. with considering the imabalancing data using oversampled data (SMOTE)
train_model(xtrain=X_train_resampled, ytrain=y_train_resampled, plot_name=f'with-SMOTE', class_weight=None)
confusion_matrix_paths=[f'./Without handling imbalance data.png',
                        f'./With-class-weight.png',
                            f'./with-SMOTE.png']
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization


## Save combined plot locally
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

## Delete old image files
for path in confusion_matrix_paths:
    os.remove(path)        
    
    
    
      ##
    
