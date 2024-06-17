import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv('/content/creditcard.csv', on_bad_lines='skip')

print(df.info())
print(df.describe())

print(df.isnull().sum())

df = df.dropna(subset=['Class'])

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

fraud_indices = np.where(y_train == 1)[0]
non_fraud_indices = np.where(y_train == 0)[0]

random_non_fraud_indices = np.random.choice(non_fraud_indices, len(fraud_indices), replace=False)
undersample_indices = np.concatenate([fraud_indices, random_non_fraud_indices])

X_train_undersample = X_train[undersample_indices]
y_train_undersample = y_train.iloc[undersample_indices]

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_undersample, y_train_undersample)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Model Performance:")
print("Precision: ", precision_score(y_test, y_pred_lr))
print("Recall: ", recall_score(y_test, y_pred_lr))
print("F1-Score: ", f1_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_undersample, y_train_undersample)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Model Performance:")
print("Precision: ", precision_score(y_test, y_pred_rf))
print("Recall: ", recall_score(y_test, y_pred_rf))
print("F1-Score: ", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()



OUTPUT:
None
                Time             V1             V2             V3  \
count  221880.000000  221880.000000  221880.000000  221880.000000   
mean    77296.823130      -0.074109      -0.011176       0.208921   
std     38551.561084       1.934175       1.652458       1.483561   
min         0.000000     -56.407510     -72.715728     -33.680984   
25%     46954.750000      -0.951293      -0.597021      -0.523647   
50%     72033.500000      -0.045141       0.070330       0.396626   
75%    118063.250000       1.236558       0.780964       1.167186   
max    142734.000000       2.454930      22.057729       9.382558   

                  V4             V5             V6             V7  \
count  221880.000000  221880.000000  221880.000000  221880.000000   
mean        0.049926      -0.071338       0.033659      -0.029730   
std         1.409439       1.372209       1.320398       1.227869   
min        -5.683171     -42.147898     -26.160506     -43.557242   
25%        -0.820037      -0.754982      -0.727724      -0.569129   
50%         0.048560      -0.130884      -0.233958       0.006476   
75%         0.846822       0.520541       0.435703       0.527879   
max        16.875344      34.801666      22.529298      36.877368   

                  V8             V9  ...            V21            V22  \
count  221880.000000  221880.000000  ...  221879.000000  221879.000000   
mean        0.005548       0.003582  ...      -0.007652      -0.027226   
std         1.207401       1.125991  ...       0.743654       0.705782   
min       -73.216718     -13.434066  ...     -34.830382     -10.933144   
25%        -0.193147      -0.662272  ...      -0.225789      -0.533715   
50%         0.034901      -0.069095  ...      -0.036414      -0.015742   
75%         0.332762       0.615891  ...       0.165608       0.463021   
max        20.007208      15.594995  ...      27.202839      10.503090   

                 V23            V24            V25            V26  \
count  221879.000000  221879.000000  221879.000000  221879.000000   
mean       -0.011342       0.001332       0.042575       0.003379   
std         0.624760       0.605510       0.504729       0.486969   
min       -44.807735      -2.836627     -10.295397      -2.604551   
25%        -0.167704      -0.347894      -0.266362      -0.331147   
50%        -0.024324       0.048476       0.081488      -0.060305   
75%         0.124381       0.425908       0.376222       0.254164   
max        19.002942       4.022866       7.519589       3.517346   

                 V27            V28         Amount          Class  
count  221879.000000  221879.000000  221879.000000  221879.000000  
mean        0.000520       0.002201      90.758764       0.001843  
std         0.399716       0.333099     250.912371       0.042895  
min       -22.565679     -11.710896       0.000000       0.000000  
25%        -0.069204      -0.046772       6.000000       0.000000  
50%         0.004004       0.016465      23.390000       0.000000  
75%         0.089147       0.078156      79.950000       0.000000  
max        12.152401      33.847808   19656.530000       1.000000  

[8 rows x 31 columns]
Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       0
V19       0
V20       0
V21       1
V22       1
V23       1
V24       1
V25       1
V26       1
V27       1
V28       1
Amount    1
Class     1
dtype: int64
<ipython-input-1-6ecc9480a117>:21: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df[col] = pd.to_numeric(df[col], errors='coerce')

Logistic Regression Model Performance:
Precision:  0.03770661157024793
Recall:  0.8902439024390244
F1-Score:  0.07234886025768086
              precision    recall  f1-score   support

         0.0       1.00      0.96      0.98     44294
         1.0       0.04      0.89      0.07        82

    accuracy                           0.96     44376
   macro avg       0.52      0.92      0.53     44376
weighted avg       1.00      0.96      0.98     44376

Random Forest Model Performance:
Precision:  0.09504685408299866
Recall:  0.8658536585365854
F1-Score:  0.1712907117008444
              precision    recall  f1-score   support

         0.0       1.00      0.98      0.99     44294
         1.0       0.10      0.87      0.17        82

    accuracy                           0.98     44376
   macro avg       0.55      0.93      0.58     44376
weighted avg       1.00      0.98      0.99     44376

