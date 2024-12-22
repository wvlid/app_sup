import numpy as np 
import pandas as pd
import sklearn.model_selection as skmodel
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
from scipy.stats import pearsonr 
import matplotlib.pyplot as plt
from numpy import ravel
from sklearn.model_selection import GridSearchCV



#Compréhension jeu de donées

features_file = "alt_acsincome_ca_features_85(1).csv"
labels_file = "alt_acsincome_ca_labels_85.csv"

df_feat = pd.read_csv(features_file)
df_lab = pd.read_csv(labels_file)
print(f"Taille des caractéristiques (features) : {df_feat.shape}")
print(f"Taille des étiquettes (labels) : {df_lab.shape}")


X_train, X_test, y_train, y_test = skmodel.train_test_split(
    df_feat, df_lab, test_size=0.2, shuffle=True, random_state=42
)

print(f"Taille de X_train : {X_train.shape}")
print(f"Taille de X_test : {X_test.shape}")

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)


# Paramètres pour le Random Forest

param_rf = {
    'n_estimators': [100, 150, 175],
    'max_depth': [15, 20, 25],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [4, 5, 6]
}

rf_model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_rf, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train.values.ravel())  # Utilisation des données d'origine

print("\nMeilleurs paramètres (Random Forest) :")
print(grid_search.best_params_)

print("\nMeilleur score (Random Forest) :")
print(grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)  # Utilisation des données d'origine pour les prédictions

print("\nMatrice de confusion (Random Forest) :")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRapport de classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nPrécision globale (Accuracy - Random Forest) : {accuracy_rf:.4f}")

filename = f'RandomForest_BestModel_{accuracy_rf:.4f}.joblib'
joblib.dump(grid_search.best_estimator_, filename)

# Paramètres pour AdaBoost
param_ada = {
    'n_estimators': [50, 75, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'algorithm': ['SAMME', 'SAMME.R']
}

ada_model = AdaBoostClassifier()
grid_search_ada = GridSearchCV(estimator=ada_model, param_grid=param_ada, cv=5, scoring='accuracy')
grid_search_ada.fit(X_train_scaled, y_train.values.ravel())  # Utilisation des données d'origine

print("\nMeilleurs paramètres (AdaBoost) :")
print(grid_search_ada.best_params_)

print("\nMeilleur score (AdaBoost) :")
print(grid_search_ada.best_score_)

best_ada_model = grid_search_ada.best_estimator_
y_pred_ada = best_ada_model.predict(X_test_scaled)  # Utilisation des données d'origine pour les prédictions

print("\nMatrice de confusion (AdaBoost) :")
print(confusion_matrix(y_test, y_pred_ada))

print("\nRapport de classification (AdaBoost) :")
print(classification_report(y_test, y_pred_ada))

accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"\nPrécision globale (Accuracy - AdaBoost) : {accuracy_ada:.4f}")

filename = f'AdaBoost_BestModel_{accuracy_ada:.4f}.joblib'
joblib.dump(grid_search_ada.best_estimator_, filename)

# Paramètres pour Gradient Boosting
param_gb = {
    'n_estimators': [100, 125, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
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

filename = f'GradientBoosting_BestModel_{accuracy_gb:.4f}.joblib'
joblib.dump(grid_search_gb.best_estimator_, filename)


base_learners = [
    ('random_forest', RandomForestClassifier(random_state=42)),
    ('gradient_boosting', GradientBoostingClassifier(random_state=42)),
    ('adaboost', AdaBoostClassifier(random_state=42)),
]

meta_learner = LogisticRegression()
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

param_grid = {
    'random_forest__n_estimators': [100, 150, 175],
    'random_forest__max_depth': [None, 10, 20],
    'adaboost__n_estimators': [50, 100, 150],
    'adaboost__learning_rate': [0.5, 1.0, 0.25],
    'gradient_boosting__n_estimators': [50, 100, 150],
    'gradient_boosting__learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train.values.ravel())

print("\nMeilleurs paramètres trouvés par GridSearchCV :")
print(grid_search.best_params_)
print("\nMeilleur score trouvé :")
print(grid_search.best_score_)

best_stacking_model = grid_search.best_estimator_

y_pred = best_stacking_model.predict(X_test_scaled)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision globale (Accuracy) : {accuracy:.4f}")

filename = f'StackingClassifier_Optimized_Model_{accuracy:.4f}.joblib'
joblib.dump(best_stacking_model, filename)
print(f"Modèle optimisé sauvegardé : {filename}")


