#Elkhatiri Walid
#Fofana Elhaji Bafodé 
#COmpréhension jeu de donées
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn.model_selection as skmodel
from sklearn import preprocessing
import joblib
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


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

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# AdaBoost
param_ada = {
    'n_estimators': [50, 75, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    'algorithm': ['SAMME', 'SAMME.R']
}

ada_model = AdaBoostClassifier()
grid_search_ada = GridSearchCV(estimator=ada_model, param_grid=param_ada, cv=5, scoring='accuracy')
grid_search_ada.fit(X_train_scaled, y_train.values.ravel())

print("\nMeilleurs paramètres (AdaBoost) :")
print(grid_search_ada.best_params_)

print("\nMeilleur score (AdaBoost) :")
print(grid_search_ada.best_score_)

best_ada_model = grid_search_ada.best_estimator_
y_pred_ada = best_ada_model.predict(X_test_scaled)

print("\nMatrice de confusion (AdaBoost) :")
print(confusion_matrix(y_test, y_pred_ada))

print("\nRapport de classification (AdaBoost) :")
print(classification_report(y_test, y_pred_ada))

accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"\nPrécision globale (Accuracy - AdaBoost) : {accuracy_ada:.4f}")

filename = f'AdaBoost_BestModel_Accuracy_{accuracy_ada:.4f}.joblib'
joblib.dump(grid_search_ada.best_estimator_, filename)

# Gradient Boosting
param_gb = {
    'n_estimators': [100, 125, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 10, 15],
    'subsample': [0.8, 0.9, 1.0]
}

gb_model = GradientBoostingClassifier()

grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train_scaled, y_train.values.ravel())

print("\nMeilleurs paramètres (Gradient Boosting) :")
print(grid_search_gb.best_params_)

print("\nMeilleur score (Gradient Boosting) :")
print(grid_search_gb.best_score_)

best_gb_model = grid_search_gb.best_estimator_

y_pred_gb = best_gb_model.predict(X_test_scaled)

print("\nMatrice de confusion (Gradient Boosting) :")
print(confusion_matrix(y_test, y_pred_gb))

print("\nRapport de classification (Gradient Boosting) :")
print(classification_report(y_test, y_pred_gb))

accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nPrécision globale (Accuracy - Gradient Boosting) : {accuracy_gb:.4f}")

filename = f'GradientBoosting_BestModel_Accuracy_{accuracy_gb:.4f}.joblib'
joblib.dump(grid_search_gb.best_estimator_, filename)
