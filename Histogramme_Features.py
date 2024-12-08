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



#AGEtab=[]



# # with open(name, newline='') as csvfile:
# #     with open("Age","w") as Age:
# #         csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
# #         for row in csvreader:
# #             Age.write(" "+row[0])
# #             AGEtab.append(row[0])
# AGEtab.pop(0)            #Pop the name of the column 
# AGEtab = list(map(float,AGEtab)) #Convert our list into list of flaot
# print(AGEtab)

#Q1
df_feat = pd.read_csv(name)
df_feat.hist()
df_lab=pd.read_csv("alt_acsincome_ca_labels_85.csv")
plt.show()

#Q2
Train_feature,Test_feature,Train_Label,Test_Label=skmodel.train_test_split(df_feat,df_lab,test_size=0.2,shuffle=True)

print(Train_feature)



#Q3/Q4
my_Scaler= preprocessing.StandardScaler()
transformer = my_Scaler.fit_transform(df_feat)
scaler_df=pd.DataFrame(transformer,columns=Train_feature.columns)

print("Standardisation")
print(transformer)
joblib.dump(my_Scaler,'scaler.joblib')

print("Standard df")
# print(scaler_df)
scaler_df.hist()
plt.show()