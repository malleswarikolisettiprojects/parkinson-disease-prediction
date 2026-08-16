#Import_Library
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#Import_Dataset
dataset = pd.read_csv('Parkinson.csv')

print('**************Dataset Head**************')
print(dataset.head())
print('**************Dataset tail**************')
print(dataset.tail())
print(dataset['status'].unique())
print('**************Datatype**************')
print(dataset.dtypes)

#The data set has 195 samples. Each row of the data set consists of voice recording of individuals with name and 23 attributes of biomedical voice measurements. The main aim of the data is to discriminate healthy people from those with Parkinson's Disease, according to "status" column which is set to 0 for healthy and 1 for individual affected with Parkinson's Disease.
parkinsons_data=dataset.drop(['name'] ,axis=1)


X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Univariate Analysis
#Univariate analysis refer to the analysis of a single variable. The main purpose of univariate analysis is to summarize and find patterns in the data. The key point is that there is only one variable involved in the analysis
#Splitting_Dataset

status_value_counts = parkinsons_data['status'].value_counts()
# print(status_value_counts)
# status_value_counts[1]

print("Number of Parkinson's Disease patients: {} ({:.2f}%)".format(status_value_counts[1], status_value_counts[1] / parkinsons_data.shape[0] * 100))
print("Number of Healthy patients: {} ({:.2f}%)".format(status_value_counts[0], status_value_counts[0] / parkinsons_data.shape[0] * 100))

sns.countplot(parkinsons_data['status'].values)
plt.xlabel("Status value")
plt.ylabel("Number of cases")
plt.show()

#Average vocal fundamental frequency MDVP:Fo(Hz)

WithDisease=parkinsons_data[parkinsons_data['status']==1]['MDVP:Fo(Hz)'].values
WithoutDisease=parkinsons_data[parkinsons_data['status']==0]['MDVP:Fo(Hz)'].values
plt.boxplot([WithDisease, WithoutDisease])
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])

sns.countplot(parkinsons_data.status, palette=['lightblue','pink'])
plt.show()

#MDVP:Fhi(Hz) - Maximum vocal fundamental frequency

diseased_freq_max = parkinsons_data[parkinsons_data["status"] == 1]["MDVP:Fhi(Hz)"].values
healthy_freq_max = parkinsons_data[parkinsons_data["status"] == 0]["MDVP:Fhi(Hz)"].values

plt.boxplot([diseased_freq_max, healthy_freq_max])
plt.title("Maximum vocal fundamental frequency MDVP:Fhi(Hz) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:Flo(Hz) - Minimum vocal fundamental frequency

diseased_vocal_min = parkinsons_data[parkinsons_data["status"] == 1]["MDVP:Flo(Hz)"].values
healthy_vocal_min = parkinsons_data[parkinsons_data["status"] == 0]["MDVP:Flo(Hz)"].values

plt.boxplot([diseased_vocal_min, healthy_vocal_min])
plt.title("MDVP:Flo(Hz) - Minimum vocal fundamental frequency Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency

diseased_percent_Jitter= parkinsons_data[parkinsons_data['status']==1]['MDVP:Jitter(%)'].values
healthy_percent_Jitter= parkinsons_data[parkinsons_data['status']==0]['MDVP:Jitter(%)'].values
plt.boxplot([diseased_percent_Jitter,healthy_percent_Jitter])
plt.title("MDVP:Jitter(%) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:Jitter(Abs)
diseased_percent_abs= parkinsons_data[parkinsons_data['status']==1]['MDVP:Jitter(Abs)'].values
healthy_percent_abs= parkinsons_data[parkinsons_data['status']==0]['MDVP:Jitter(Abs)'].values
plt.boxplot([diseased_percent_Jitter,healthy_percent_Jitter])
plt.title("MDVP:Jitter(Abs) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:RAP

diseased_rap_mdvp= parkinsons_data[parkinsons_data['status']==1]['MDVP:RAP'].values
healthy_rap_mdvps= parkinsons_data[parkinsons_data['status']==0]['MDVP:RAP'].values
plt.boxplot([diseased_rap_mdvp,healthy_rap_mdvps])
plt.title("MDVP:RAP) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:PPQ
disease_mdvp_ppq = parkinsons_data[parkinsons_data['status']==1]['MDVP:PPQ'].values
healthy_mdvp_ppq =parkinsons_data[parkinsons_data['status']==0]['MDVP:PPQ'].values
plt.boxplot([disease_mdvp_ppq, healthy_mdvp_ppq])
plt.title("MDVP:PPQ) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#Jitter:DDP
disease_jitterp_ddp = parkinsons_data[parkinsons_data['status']==1]['Jitter:DDP'].values
healthy_jitter_ddp =parkinsons_data[parkinsons_data['status']==0]['Jitter:DDP'].values
plt.boxplot([disease_jitterp_ddp, healthy_jitter_ddp])
plt.title("Jitter:DDP) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude

disease_mdvp_shimmer = parkinsons_data[parkinsons_data['status']==1]['MDVP:Shimmer']
healthy_mdvp_shimmer = parkinsons_data[parkinsons_data['status']==0]['MDVP:Shimmer']
plt.boxplot([disease_mdvp_shimmer , healthy_mdvp_shimmer])
plt.title("Jitter:DDP) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()

#Bivariate Analysis

#sns.pairplot(parkinsons_data, hue="status", diag_kind='kde')
#plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier
rdF=RandomForestClassifier(n_estimators=250, max_depth=50,random_state=45)
rdF.fit(X_train,y_train)
pred = rdF.predict(X_test)
accuracy_RF = accuracy_score(pred, y_test)*100
cmrf=confusion_matrix(y_test, pred)
print("1.Random Forest Accuracy")
print(accuracy_RF)
print("Random Forest classification_report")
print(classification_report(pred, y_test, labels=None))
print("Random Forest confusion_matrix")
print(cmrf)


from sklearn import svm
from sklearn import tree

# DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
cmdt=confusion_matrix(y_test, pred)
# accuracy
accuracy_DT = accuracy_score(pred, y_test)*100
print("2.Decision Tree Accuracy")
print(accuracy_DT)
print("Decision Tree  classification_report")
print(classification_report(pred, y_test, labels=None))
print("Decision Tree  confusion_matrix")
print(cmdt)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Naive Bayes algorithm
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# pred
pred = gnb.predict(X_test)
cmnb=confusion_matrix(y_test, pred)
# accuracy
accuracy_NB = accuracy_score(pred, y_test)*100
print("3. Naive Bayes Accuracy")
print(accuracy_NB)
print("Naive Bayes classification_report")
print(classification_report(pred, y_test, labels=None))
print("Naive Bayes confusion_matrix")
print(cmnb)



#KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
pred = neigh.predict(X_test)
cmknn=confusion_matrix(y_test, pred)
 # accuracy
accuracy_KNN = accuracy_score(pred, y_test)*100

print("4. KNeighborsClassifier Accuracy")
print(accuracy_KNN)
print("KNeighborsClassifier classification_report")
print(classification_report(pred, y_test, labels=None))
print("KNeighborsClassifier confusion_matrix")
print(cmknn)


#LinearSVM
from sklearn import svm
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
pred = lin_clf.predict(X_test)
cmsvm=confusion_matrix(y_test, pred)
# accuracy
accuracy_SVM = accuracy_score(pred, y_test)*100
print("5. svm Accuracy")
print(accuracy_SVM)
print("svm classification_report")
print(classification_report(pred, y_test, labels=None))
print("svm confusion_matrix")
print(cmsvm)


#XGBClassifier
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)
pred=xgb_clf.predict(X_test)
cmxg=confusion_matrix(y_test, pred)

accuracy_XGB = accuracy_score(y_test,pred)*100
print("6. XGBClassifier Accuracy")
print(accuracy_XGB)
print("XGBClassifier classification_report")
print(classification_report(pred, y_test, labels=None))
print("XGBClassifier confusion_matrix")
print(cmxg)

# plot_confusion_matrix of rf
array =  [[13,0],
          [2,44]]
df_confusion_matrix = pd.DataFrame(array, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (2,2))
sns.heatmap(df_confusion_matrix, annot=True)
plt.title('Confusion matrix of XGBClassifier  ', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
pred=clf.predict(X_test)
cmlr=confusion_matrix(y_test, pred)

accuracy_LR = accuracy_score(y_test,pred)*100
print("7. Logistic Regression Accuracy")
print(accuracy_LR)
print("Logistic Regression classification_report")
print(classification_report(pred, y_test, labels=None))
print("Logistic Regression confusion_matrix")
print(cmlr)

labels = ["RF","DT ", "NB", "KNN", "SVM", "XGB", "LR"]
usages = [accuracy_RF,accuracy_DT,accuracy_NB,accuracy_KNN,accuracy_SVM,accuracy_XGB,accuracy_LR]

y_positions = range(len(labels))
plt.bar(y_positions,usages)
plt.xticks(y_positions, labels)
plt.ylabel("Accuracy")
plt.title("model selection")
plt.show()

# Saving model to disk
pickle.dump(xgb_clf, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
arr = np.array([[237.226,247.326,225.227,0.00298,0.00001,0.00169,0.00182,0.00507,0.01752,0.164,0.01035,0.01024,0.01133,0.03104,0.0074,22.736,0.305062,0.654172,-7.31055,0.098648,2.416838,0.095032]])
print(model.predict(arr))
print(model.predict_proba(arr))