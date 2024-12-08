#Elkhatiri Walid
#Fofana Elhaji Bafodé 
#COmpréhension jeu de donées
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn.model_selection as skmodel
from sklearn import preprocessing
import joblib
#path=""

name="alt_acsincome_ca_features_85(1).csv"





#Q1
features_file = "alt_acsincome_ca_features_85(1).csv"
labels_file = "alt_acsincome_ca_labels_85.csv"

df_feat = pd.read_csv(features_file)
df_lab = pd.read_csv(labels_file)
print(f"Taille des caractéristiques (features) : {df_feat.shape}")
print(f"Taille des étiquettes (labels) : {df_lab.shape}")

df_feat.hist(figsize=(10, 8))


#Q2
X_train, X_test, y_train, y_test = skmodel.train_test_split(
    df_feat, df_lab, test_size=0.2, shuffle=True, random_state=42
)

print(f"Taille de X_train : {X_train.shape}")
print(f"Taille de X_test : {X_test.shape}")

#Q3/Q4
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')

scaler_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
scaler_df.hist(figsize=(10, 8), bins=20)

from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, classification_report


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train.values.ravel())  # Rendre y_train compatible
# n_estimators=100, max_depth=4, min_samples_split= 2, min_samples_leaf=2,


y_pred = rf_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

importances = rf_model.feature_importances_
for feature, importance in zip(df_feat.columns, importances):
    print(f"{feature}: {importance:.4f}")
