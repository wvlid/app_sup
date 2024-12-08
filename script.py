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

param_rf = {
    'n_estimators': [100, 150, 175, 200, 250],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [8, 10, 12, 15],
    'min_samples_leaf': [4, 5, 6, 7]
}


rf_model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_rf, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train.values.ravel())

print("\nMeilleurs paramètres (Random Forest) :")
print(grid_search.best_params_)

print("\nMeilleur score (Random Forest) :")
print(grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)

print("\nMatrice de confusion (Random Forest) :")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRapport de classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nPrécision globale (Accuracy - Random Forest) : {accuracy_rf:.4f}")

# Enregistrement du modèle avec l'accuracy dans le nom du fichier
filename = f'RandomForest_BestModel_Accuracy_{accuracy_rf:.4f}.joblib'
joblib.dump(grid_search.best_estimator_, filename)


