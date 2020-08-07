#Basic Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

#Data Pre-processing
from sklearn.preprocessing import StandardScaler

#Model Training and Validation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, classification_report, plot_confusion_matrix

#ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier

#Model Export
import joblib
from joblib import dump #from joblib import load > to load .pkl file

df = pd.read_csv('cardio_train.csv', sep=';')
df.shape

df.head()

df_copy = df.copy() # Create a copy as backup
df_copy.head(2)

df.info()

df.describe()

df.columns

print('>>>> # of Rows ::', df.shape[0])
print('>>>> # of Columns ::', df.shape[1])
print('>> # of unique values in df <<')
for i in df:
    print(i, '--->', df[i].nunique())
    
[i for i in df if df[i].nunique()<5]

for i in df.columns:
    if df[i].nunique()<5:
        print('#######', i, '#######')
        print(df[i].unique())
        
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull())

# =============================================================================
# Observations:
#     1. No null values in the dataset.
#     2. All features are numerical features including Age (in days).
#     3. Huge outliers in 'ap_hi' and 'ap_lo'. We will do further analysis to understand these outliers.
#     4. 'id' is unique and can be dropped.
# =============================================================================

#Univariate analysis of numerical features
for i in df:
    f=plt.figure(figsize=(17,3))
    f.add_subplot(1,2,1)
    sns.distplot(df[i],bins=30)
    f.add_subplot(1,2,2)
    sns.boxplot(df[i])
    print('Skewness of %s = %0.2f' %(i, df[i].skew()))
    
# =============================================================================
# Observations:
#     1. 'cholestrol', 'gluc', 'smoke' and 'alco' is slightly skewed to the right.
#     2. Majority of the data is non-smokers and non-alcoholic.
#     3. We have few age values below 14000 days.
#     4. 'ap_hi' and 'ap_lo' above 4000 needs to be analyzed and will be removed if it has no significant contribution.
# =============================================================================

col_lst = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
for i in col_lst:
    plt.figure(figsize=(17,4))
    sns.countplot(x=i, data=df, hue='cardio')

sns.countplot(x='cardio', data=df)
df['cardio'].value_counts()

df[df['age']<14000]
# Four outliers that belong to 'cardio'=0. We can remove them as they are outliers and doesn't affect our target count.
df = df[df['age']>14000]

df[df['ap_hi']>4000]['cardio'].value_counts()
df = df[df['ap_hi']<4000]

df[df['ap_lo']>4000]['cardio'].value_counts()
df = df[df['ap_lo']<4000]

df.shape

# Correlation Analysis
plt.figure(figsize=(12,5))
sns.heatmap(df.corr(), vmax=1, vmin=-1, annot= True, fmt='.1g', cmap='bwr', mask=np.triu(df.corr()))

# Dropping unwanted columns
df.drop(columns=['id'], axis=1, inplace=True)
df.shape

sns.pairplot(df)

x = df.drop(columns=['cardio'], axis=1)
y = df['cardio']

print('X >>\n', x)
print('Y >>\n', y)

# Instantiate and scale the data
ss = StandardScaler()
x = ss.fit_transform(x)

### Machine Learning Algorithms
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('x_train.shape -->',x_train.shape,'| x_test.shape -->',x_test.shape,'\ny_train.shape -->',y_train.shape,'| y_test.shape -->',y_test.shape)

# Instantiate all models
logr = LogisticRegression()
gnb = GaussianNB()
svc = SVC()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
xgb = XGBClassifier()
rf = RandomForestClassifier()
bgc = BaggingClassifier()
etc = ExtraTreesClassifier()
gbc = GradientBoostingClassifier()

models={'Logistic Regression':logr,
        'Gaussian NB':gnb,
        'Support Vector Machine':svc,
        'KNeighbors Classifier':knn,
        'Decision Tree Classifier':dtc,
        'XGB Classifier':xgb,
        'Random Forest Classifier':rf,
        'Bagging Classifier':bgc,
        'Extra Trees Classifier':etc,
        'Gradient Boosting Classifier':gbc
       }

# Create a function to run all models
def main(cls):
    a_scores = []
    for model_name, model in cls.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        a_scores.append(score*100)
        print('##############################',model_name,'##############################')
        print('>>> Accuracy Score = %0.2f' %(score*100))
        c_matrix = confusion_matrix(y_test, y_pred)
        print('>>> Confusion Matrix: \n', c_matrix)
        TN, FP, FN, TP = c_matrix[0,0], c_matrix[0,1], c_matrix[1,0], c_matrix[1,1]
        print('>>> Recall Score = %0.2f' %((TP)*100/float(TP+FN)))
        print('>>> Specificity = %0.2f' %(TN*100/float(TN+FP)))
        print('>>> False Positive Rate = %0.2f' %(FP*100/float(FP+TN)))
        print('>>> Precision Score = %0.2f' %(TP*100/float(TP+FP)))
        print('>>> Classification Report:')
        print(classification_report(y_test, y_pred))
    return a_scores

a_scores = main(models)

p = pd.DataFrame(data=a_scores, columns=['Accuracy Score'], index=list(models.keys())).sort_values(by = 'Accuracy Score', ascending=False)
print(p)

# Let's take the top three models + Logistic Regression and perform cross validation.
new_models = {'Gradient Boosting Classifier':gbc,
              'Support Vector Machine':svc,
              'XGB Classifier':xgb,
              'Logistic Regression':logr
              }

cv_scores = []
for model_name, model in new_models.items():
    score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=10)
    cv_scores.append((score.mean())*100)
    print(model_name,' >>> Completed')
print(cv_scores)

pd.DataFrame(data=cv_scores, columns=['New Accuracy Score'], index=list(new_models.keys())).join(p)

### Hyperparameter Tuning
# Instantiating with default values
gbc = GradientBoostingClassifier(random_state=42)
svc = SVC()
xgb = XGBClassifier()
logr = LogisticRegression(random_state=42)

### GridSearchCV Hyperparameter Tuning
gbc_param = {'n_estimators':range(20,81,10)}
b_gbc=GridSearchCV(gbc,gbc_param)
b_gbc.fit(x_train,y_train)
print('Gradient Boosting Classifier >>>', b_gbc.best_params_)

svc_param={'kernel':['linear','poly','rbf'],'C':[1,10]}
b_svc=GridSearchCV(svc,svc_param)
b_svc.fit(x_train,y_train)
print('Support Vector Machine >>>', b_svc.best_params_)

xgb_param = {'learning_rate':[0.1,1],
             'n_estimators':range(50,251,50)
             }
b_xgb=GridSearchCV(xgb,xgb_param)
b_xgb.fit(x_train,y_train)
print('XGB Classifier >>>', b_xgb.best_params_)

logr_param = {'C':[0.1,1,10]}
b_logr=GridSearchCV(logr,logr_param)
b_logr.fit(x_train,y_train)
print('Logistic Regression >>>', b_logr.best_params_)

# Instantiating with new parameters
gbc = GradientBoostingClassifier(random_state=42, n_estimators=80)
svc = SVC(C=1, kernel='linear')
xgb = XGBClassifier(learning_rate=0.1, n_estimators=150)
logr = LogisticRegression(random_state=42, C=0.1)

new_models = {'Gradient Boosting Classifier':gbc,
              'Support Vector Machine':svc,
              'XGB Classifier':xgb,
              'Logistic Regression':logr
              }

a_scores = []
for model_name, model in new_models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test,y_pred)
    a_scores.append(score*100)
    print('>>>', model_name, 'Accuracy Score = %0.2f' %(score*100))

pd.DataFrame(data=a_scores, columns=['New Accuracy Score'], index=list(new_models.keys())).join(p)

# Plotting Confusion Matrix
gbc.fit(x_train, y_train)
plot_confusion_matrix(gbc, x_test, y_test)

# Plotting ROC Curve for the model
y_pred_prob = gbc.predict_proba(x_test)[:, 1]  #1 is the probabilty of threshold value
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Gradient Boosting Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gradient Boosting Classifier')
plt.show()

#Exporting model as pkl file
joblib.dump(gbc,'GBC_Cardiovascular_Disease_Detection.pkl')