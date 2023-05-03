# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from imblearn.metrics import specificity_score
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

location = pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_with_Counties_and_FIPS.pickle')
location = location.drop(columns = ['FIPS','county_lat','county_long'])

"""# General Functions

## Class Imbalance
"""

import pandas as pd

def check_imbalance(df, threshold=0.2):
    imbalance_data = []

    for col in df.columns:
        col_counts = df[col].value_counts()
        col_percentages = col_counts / col_counts.sum()
        minority_percentage = col_percentages.min()
        if minority_percentage < threshold:
            unbalanced = 'Yes'
        else:
            unbalanced = 'No'

        imbalance_data.append({
            'Column Name': col,
            'Minority Class': col_percentages.idxmin(),
            'Minority Percentage': minority_percentage * 100,
            'Is Unbalanced': unbalanced
        })

    
    return pd.DataFrame(imbalance_data)

"""## Plot Confusion Matrix"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm,classes,Title):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)


    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'{Title} Confusion Matrix')
    
    plt.show()

"""## Plot AUC"""

def displayAUC(y_test_binarized_best, y_score_best, class_names, title):
    n_classes = y_test_binarized_best.shape[1]
    
    if n_classes == 1:
        fpr, tpr, _ = roc_curve(y_test_binarized_best, y_score_best)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr))
        fig.update_layout(
            title=f"{title} (AUC-ROC = {roc_auc:.6f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
    else: # multi-class classification

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized_best[:, i], y_score_best[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        auc_roc = roc_auc_score(y_test_binarized_best, y_score_best, multi_class='ovr', average='weighted')


        fig = go.Figure()

        for i in range(n_classes):
            fig.add_trace(
                go.Scatter(
                    x=fpr[i],
                    y=tpr[i],
                    mode="lines",
                    name=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
                )
            )
        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=0, y1=1,
            yref="y", xref="x",
            line=dict(color="rgba(0, 0, 0, 0.3)", dash="dash")
        )

        fig.update_layout(
            title=f"{title} (Weighted AUC-ROC = {auc_roc:.6f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend_title="Class",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

    return fig

"""## Descision Tree"""

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

def runModel_dt(classifier, X, y, n_iterations=10, save_model=True, model_filename='Decision_Tree.pkl',displayOutput=True):
    feature_importances = np.zeros(X.shape[1])
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_roc_scores = []
    best_auc_roc = -1
    best_model = None
    best_scores = None
    y_test_binarized_best = None
    y_score_best= None
    specificity_scores = []
    balanced_accuracy_scores = []
    best_balanced_accuracy = -1
    

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    
    tqdm.pandas(desc = 'Iterations')
    for i in tqdm(range(n_iterations)):
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        

            classifier.fit(X_train, y_train)

            feature_importances += classifier.feature_importances_


            y_pred = classifier.predict(X_test)
            y_score = classifier.predict_proba(X_test)[:, 1]
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_score)
            specificity = specificity_score(y_test,y_pred,average='weighted')
            balanced_accuracy = (recall + specificity) / 2

            if balanced_accuracy > best_balanced_accuracy:
              best_auc_roc = auc_roc
              best_model = classifier
              y_test_binarized_best = y_test_binarized
              y_score_best= y_score
              best_specificity = specificity
              best_balanced_accuracy  = balanced_accuracy
              best_scores = (accuracy, precision, recall, f1, auc_roc,best_specificity,best_balanced_accuracy)

            cm = confusion_matrix(y_test, y_pred)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_roc_scores.append(auc_roc)
            specificity_scores.append(specificity)
            balanced_accuracy_scores.append(balanced_accuracy)

            if displayOutput == True:
              print(f"Iteration {i + 1}:")
              print(f"  Accuracy: {accuracy:.2f}")
              print(f"  Precision: {precision:.2f}")
              print(f"  Recall: {recall:.2f}")
              print(f"  F1 score: {f1:.2f}")
              print(f"  AUC-ROC score: {auc_roc:.2f}")
              print(f"  Specificity score: {specificity:.2f}")
              print(f"  Balanced Accuracy Score: {balanced_accuracy:.2f}")

    with open(model_filename, 'wb') as f:
      pickle.dump(best_model, f)

    average_importances = feature_importances / n_iterations
    average_accuracy = np.mean(accuracy_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)
    average_auc_roc = np.mean(auc_roc_scores)
    average_specificity = np.mean(specificity_scores)
    average_balance = np.mean(balanced_accuracy_scores)

    return cm,(average_accuracy, average_precision, average_recall, average_f1, average_auc_roc,average_specificity,average_balance), (accuracy, precision, recall, f1, auc_roc,best_specificity,best_balanced_accuracy), best_model, y_test_binarized_best, y_score_best

"""## Random Forest"""

def runModel_rf(classifier, X, y, n_iterations=10, save_model=True, model_filename='Random_Forest.pkl', displayOutput=True):
    feature_importances = np.zeros(X.shape[1])
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_roc_scores = []
    specificity_scores = []
    balanced_accuracy_scores = []
    best_auc_roc = -1
    best_balanced_accuracy = -1
    best_model = None
    best_scores = None
    y_test_binarized_best = None
    y_score_best = None

    tqdm.pandas(desc = 'Iterations')
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    for i in tqdm(range(n_iterations)):
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            classifier.fit(X_train, y_train)


            feature_importances += classifier.feature_importances_


            y_pred = classifier.predict(X_test)
            y_score = classifier.predict_proba(X_test)[:,1]
            y_test_binarized = label_binarize(y_test, classes=[0, 1])
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary')        
            specificity = specificity_score(y_test,y_pred,average='binary')
            balanced_accuracy = (recall + specificity) / 2

            try:
                auc_roc = roc_auc_score(y_test_binarized, y_score, average='weighted')
            except ValueError:
                print(f"Iteration {i + 1}: Skipped due to only one class present in y_true.")
                continue

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_auc_roc = auc_roc
                best_model = classifier
                best_scores = (accuracy, precision, recall, f1, auc_roc, specificity, balanced_accuracy)
                y_test_binarized_best = y_test_binarized
                y_score_best = y_score

            cm = confusion_matrix(y_test, y_pred)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_roc_scores.append(auc_roc)
            specificity_scores.append(specificity)
            balanced_accuracy_scores.append(balanced_accuracy)

            if displayOutput == True:
                print(f"Iteration {i + 1}:")
                print(f"  Accuracy: {accuracy:.2f}")
                print(f"  Precision: {precision:.2f}")
                print(f"  Recall: {recall:.2f}")
                print(f"  F1 score: {f1:.2f}")
                print(f"  AUC-ROC score: {auc_roc:.2f}")
                print(f"  Specificity score: {specificity:.2f}")
                print(f"  Balanced Accuracy Score: {balanced_accuracy:.2f}")

    with open('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/randomForest.pkl', 'wb') as f:
      pickle.dump(best_model, f)


    average_importances = feature_importances / n_iterations
    average_accuracy = np.mean(accuracy_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)
    average_auc_roc = np.mean(auc_roc_scores)
    average_specificity = np.mean(specificity_scores)
    average_balance = np.mean(balanced_accuracy_scores)

    return cm,average_importances, (average_accuracy, average_precision, average_recall,  average_f1, average_auc_roc, average_specificity,average_balance), best_scores,best_model,y_test_binarized_best,y_score_best

"""## KNN"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import pickle

def runModel_KNN(classifier, X, y, n_iterations=10, save_model=True, model_filename='KNN.pkl', displayOutput=True):
  accuracy_scores = []
  precision_scores = []
  recall_scores = []
  f1_scores = []
  auc_roc_scores = []
  specificity_scores = []
  balanced_accuracy_scores = []
  best_auc_roc = -1
  best_model = None
  best_scores = None
  y_test_binarized_best = None
  y_score_best= None
  best_balanced_accuracy = -1
  cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
  tqdm.pandas(desc = 'Iterations')

  for i in tqdm(range(n_iterations)):
    for train_index, test_index in cv.split(X, y):
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]
      

          classifier.fit(X_train, y_train)


          y_pred = classifier.predict(X_test)
          y_score = classifier.predict_proba(X_test)[:,1]
          y_test_binarized = label_binarize(y_test, classes=[0,1])
          accuracy = accuracy_score(y_test, y_pred)
          precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
          recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
          f1 = f1_score(y_test, y_pred, average='binary')
          try:
              auc_roc = roc_auc_score(y_test_binarized, y_score, average='weighted')
          except ValueError:
              print(f"Iteration {i + 1}: Skipped due to only one class present in y_true.")
              continue
          specificity = specificity_score(y_test,y_pred)
          balanced_accuracy = (recall + specificity) / 2

          if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_auc_roc = auc_roc
          best_model = classifier
          best_scores = (accuracy, precision, recall, f1, auc_roc,specificity,balanced_accuracy)
          y_test_binarized_best = y_test_binarized
          y_score_best= y_score

          accuracy_scores.append(accuracy)
          precision_scores.append(precision)
          recall_scores.append(recall)
          f1_scores.append(f1)
          auc_roc_scores.append(auc_roc)
          specificity_scores.append(specificity)
          balanced_accuracy_scores.append(balanced_accuracy)

          cm = confusion_matrix(y_test, y_pred)


          if displayOutput == True:
              print(f"Iteration {i + 1}:")
              print(f"  Accuracy: {accuracy:.2f}")
              print(f"  Precision: {precision:.2f}")
              print(f"  Recall: {recall:.2f}")
              print(f"  F1 score: {f1:.2f}")
              print(f"  AUC-ROC score: {auc_roc:.2f}")
              print(f"  Specificity score: {specificity:.2f}")
              print(f"  Balanced Accuracy Score: {balanced_accuracy:.2f}")

  with open(model_filename, 'wb') as f:
      pickle.dump(best_model, f)

  average_accuracy = np.mean(accuracy_scores)
  average_precision = np.mean(precision_scores)
  average_recall = np.mean(recall_scores)
  average_f1 = np.mean(f1_scores)
  average_auc_roc = np.mean(auc_roc_scores)
  average_specificity = np.mean(specificity_scores)
  average_balance = np.mean(balanced_accuracy_scores)

  avg_scores = (average_accuracy, average_precision, average_recall, average_f1, average_auc_roc, average_specificity, average_balance)

  return cm, avg_scores, best_scores, best_model, y_test_binarized_best, y_score_best

"""# Global Warming Belief

## Target and Training data
"""

target = 'glbcc'
data, holdout = train_test_split(location, test_size=0.2, stratify=location[target])
data = data.reset_index()


X = data.drop(target, axis=1)
y = data[target]
class_names  = ['Yes','No']

"""#"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

threshold = .05

info_gain = mutual_info_classif(X_encoded, y)

fig, ax = plt.subplots()
ax.bar(X.columns, info_gain)
ax.axhline(y=threshold, color='r', linestyle='--')
ax.set_xticklabels(X.columns, rotation=90)
ax.set_ylabel('Information gain')
ax.set_title('How much risk do you think global warming poses for people and the environment?')
plt.show()


selected_cols = X.columns[info_gain > threshold].tolist()
print(f'Selected columns: {selected_cols}')

"""#"""

df = data[selected_cols]

threshold = 0.2
imbalance_df = check_imbalance(df, threshold)
print(imbalance_df)

"""## Train test split

---


"""



X = data[selected_cols]
X_encoded = encoder.fit_transform(X)

"""## Decision Tree

### Model
"""

:
from imblearn.metrics import specificity_score
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_features='sqrt')
cm,(average_accuracy, average_precision, average_recall, average_f1, average_auc_roc,average_specificity,average_balance), (best_accuracy, best_precision, best_recall, best_f1, best_auc_roc,best_specificity,best_balance),best_model, y_test_binarized_best, y_score_best = runModel_dt(dt_classifier, X_encoded, y, n_iterations=1,displayOutput=True)

DecisionTree = export_text(best_model, feature_names=list(X.columns), decimals=2)
Scores = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC-ROC','Specificity','Balanced Accuracy'],
    'Average': [average_accuracy, average_precision, average_recall, average_f1, average_auc_roc,average_specificity,average_balance],
    'Best': [best_accuracy, best_precision, best_recall, best_f1, best_auc_roc,best_specificity,best_balance],
}
bestVisualization = displayAUC(y_test_binarized_best,y_score_best,class_names,title = 'Best model')

"""### Holdout validation"""

holdout_test = holdout[target]
holdout1 = holdout.drop(target, axis=1)
holdout1 = holdout1[selected_cols]
holdout1 = encoder.fit_transform(holdout1)



y_pred = best_model.predict(holdout1)
y_score = best_model.predict_proba(holdout1)[:, 1]
y_test_binarized = label_binarize(holdout_test, classes=np.unique(y))
accuracy = accuracy_score(holdout_test, y_pred)
precision = precision_score(holdout_test, y_pred)
recall = recall_score(holdout_test, y_pred)
f1 = f1_score(holdout_test, y_pred)
auc_roc = roc_auc_score(holdout_test, y_score)
specificity = specificity_score(holdout_test,y_pred,average='weighted')
balanced_accuracy = (recall + specificity) / 2
Scores['Holdout'] = [accuracy, precision, recall, f1, auc_roc,specificity,balanced_accuracy]


holdout_visualization = displayAUC(y_test_binarized,y_score,class_names,title = 'Holdout')

cm2 = confusion_matrix(holdout_test, y_pred)

"""### Display All output"""

scoresDF = pd.DataFrame(Scores)
scoresDF.set_index('Metric', inplace=True)
scoresDF['Difference'] = scoresDF['Best'] - scoresDF['Holdout']
scoresDF['Sign'] = scoresDF['Difference'].apply(lambda x: '+' if x >= 0 else '-')
display(scoresDF)

plot_confusion_matrix(cm,class_names,'Best')
bestVisualization.show()

plot_confusion_matrix(cm2,class_names,'Holdout')
holdout_visualization.show()

"""## Random Forest

### Grid Search
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)

base_classifier = RandomForestClassifier(criterion='entropy', max_depth=6, max_features='sqrt',
                                  bootstrap=True, class_weight='balanced', min_samples_leaf=10)


param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 6, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}


from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=50)
grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1)


grid_search.fit(X_train, y_train)


print("Best parameters: ", grid_search.best_params_)
print("Best Balanced Accuracy: ", grid_search.best_score_)

best_classifier = grid_search.best_estimator_

"""### Model"""

rf_classifier = RandomForestClassifier(criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=5,n_estimators=10,min_samples_split=2)
cm,average_importances, (average_accuracy, average_precision, average_recall, average_f1, average_auc_roc,average_specificity,average_balance), (best_accuracy, best_precision, best_recall, best_f1, best_auc_roc,best_specificity,best_balance),best_model,y_test_binarized_best,y_score_best= runModel_rf(rf_classifier, X_encoded, y, n_iterations=100,displayOutput=False)

Scores = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC-ROC','Specificity','Balanced Accuracy'],
    'Average': [average_accuracy, average_precision, average_recall, average_f1, average_auc_roc,average_specificity,average_balance],
    'Best': [best_accuracy, best_precision, best_recall, best_f1, best_auc_roc,best_specificity,best_balance],
}

bestVisualization = displayAUC(y_test_binarized_best,y_score_best,class_names,title = 'Best model')


"""### Holdout Validation"""

holdout_test = holdout[target]
holdout1 = holdout.drop(target, axis=1)
holdout1 = holdout1[selected_cols]
holdout1 = encoder.fit_transform(holdout1)



y_pred = best_model.predict(holdout1)
y_score = best_model.predict_proba(holdout1)[:, 1]
y_test_binarized = label_binarize(holdout_test, classes=np.unique(y))
accuracy = accuracy_score(holdout_test, y_pred)
precision = precision_score(holdout_test, y_pred)
recall = recall_score(holdout_test, y_pred)
f1 = f1_score(holdout_test, y_pred)
auc_roc = roc_auc_score(holdout_test, y_score)
specificity = specificity_score(holdout_test,y_pred,average='weighted')
balanced_accuracy = (recall + specificity) / 2
Scores['Holdout'] = [accuracy, precision, recall, f1, auc_roc,specificity,balanced_accuracy]

holdout_visualization = displayAUC(y_test_binarized,y_score,class_names,title = 'Holdout')

cm2 = confusion_matrix(holdout_test, y_pred)

"""### Display Visualziations"""

scoresDF = pd.DataFrame(Scores)
scoresDF.set_index('Metric', inplace=True)
scoresDF['Difference'] = scoresDF['Best'] - scoresDF['Holdout']
scoresDF['Sign'] = scoresDF['Difference'].apply(lambda x: '+' if x >= 0 else '-')
display(scoresDF)

plot_confusion_matrix(cm,class_names,'Best')
bestVisualization.show()

plot_confusion_matrix(cm2,class_names,'Holdout')
holdout_visualization.show()

"""## KNN

### Grid search
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)


base_classifier = KNeighborsClassifier()


param_grid = {
    'n_neighbors': [5,7,9,11],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski','euclidean','manhattan']
}


cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=50)
grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)


grid_search.fit(X_train, y_train)


print("Best parameters: ", grid_search.best_params_)
print("Best balanced accuracy: ", grid_search.best_score_)


best_classifier = grid_search.best_estimator_

"""### Model"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

knn_classifier = KNeighborsClassifier(n_neighbors=11,metric='minkowski',weights='uniform')
cm,avgScores, best_scores ,best_model, y_test_binarized_best, y_score_best = runModel_KNN(knn_classifier, X_encoded, y, n_iterations=100, displayOutput=False)

Scores = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC-ROC', 'Specificity', 'Balanced Accuracy'],
    'Average': [average_accuracy, average_precision, average_recall, average_f1, average_auc_roc, average_specificity, average_balance],
    'Best': [best_accuracy, best_precision, best_recall, best_f1, best_auc_roc, best_specificity, best_balance],
}


bestVisualization = displayAUC(y_test_binarized_best,y_score_best,class_names,title = 'Best model')

"""### Holdout"""

holdout_test = holdout[target]
holdout1 = holdout.drop(target, axis=1)
holdout1 = holdout1[selected_cols]
holdout1 = encoder.fit_transform(holdout1)


y_test_binarized = label_binarize(holdout_test, classes=[0, 1])
y_score = best_model.predict_proba(holdout1)[:, 1]
y_pred = best_model.predict(holdout1)


accuracy = accuracy_score(holdout_test, y_pred)
precision = precision_score(holdout_test, y_pred, pos_label=1, average='binary', zero_division=0)
recall = recall_score(holdout_test, y_pred, pos_label=1, average='binary', zero_division=0)
f1 = f1_score(holdout_test, y_pred, pos_label=1, average='binary')

auc_roc = roc_auc_score(y_test_binarized, y_score, average='weighted')


tn, fp, fn, tp = confusion_matrix(holdout_test, y_pred).ravel()
specificity = tn / (tn + fp)
balanced_accuracy = (recall + specificity) / 2


Scores['Holdout'] = [accuracy, precision, recall, f1, auc_roc, specificity, balanced_accuracy]

holdout_visualization = displayAUC(y_test_binarized, y_score, ['Negative', 'Positive'], title='Holdout')


cm2 = confusion_matrix(holdout_test, y_pred)

"""### Scores"""

scoresDF = pd.DataFrame(Scores)
scoresDF.set_index('Metric', inplace=True)
scoresDF['Difference'] = scoresDF['Best'] - scoresDF['Holdout']
scoresDF['Sign'] = scoresDF['Difference'].apply(lambda x: '+' if x >= 0 else '-')
display(scoresDF)

plot_confusion_matrix(cm,class_names,'Best')
bestVisualization.show()

plot_confusion_matrix(cm2,class_names,'Best')
holdout_visualization.show()

"""## CatBoost

### Model
"""

!pip install catboost

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier


param_grid = {
    'iterations': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.5],
    'depth': [4, 6, 8],
    'loss_function': ['Logloss'],
}


catboost_classifier = CatBoostClassifier()


grid_search = GridSearchCV(catboost_classifier, param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_encoded, y)


print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

from catboost import CatBoostClassifier


def runModel_CatBoost(classifier, X, y, n_iterations=10, save_model=True, model_filename='CatBoost.pkl', displayOutput=True):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_roc_scores = []
    specificity_scores = []
    balanced_accuracy_scores = []
    best_auc_roc = -1
    best_model = None
    best_scores = None
    y_test_binarized_best = None
    y_score_best = None
    best_balanced_accuracy = -1


    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3)

    for i in range(n_iterations):
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            classifier.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)


            y_pred = classifier.predict(X_test)
            y_score = classifier.predict_proba(X_test)[:, 1]
            y_test_binarized = label_binarize(y_test, classes=[0, 1])
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary')
            specificity = specificity_score(y_test, y_pred, average='binary')
            balanced_accuracy = (recall + specificity) / 2

            try:
                auc_roc = roc_auc_score(y_test_binarized, y_score, average='weighted')
            except ValueError:
                print(f"Iteration {i + 1}: Skipped due to only one class present in y_true.")
                continue

            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_auc_roc = auc_roc
                best_model = classifier
                best_scores = (accuracy, precision, recall, f1, auc_roc, specificity, balanced_accuracy)
                y_test_binarized_best = y_test_binarized
                y_score_best = y_score

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            auc_roc_scores.append(auc_roc)
            specificity_scores.append(specificity)
            balanced_accuracy_scores.append(balanced_accuracy)

            cm = confusion_matrix(y_test, y_pred)

            if displayOutput == True:
                print(f"Iteration {i + 1}:")
                print(f"  Accuracy: {accuracy:.2f}")
                print(f"  Precision: {precision:.2f}")
                print(f"  Recall: {recall:.2f}")
                print(f"  F1 score: {f1:.2f}")
                print(f"  AUC-ROC score: {auc_roc:.2f}")
                print(f"  Specificity score: {specificity:.2f}")
                print(f"  Balanced Accuracy Score: {balanced_accuracy:.2f}")

    with open(model_filename, 'wb') as f:
      pickle.dump(best_model, f)
    average_accuracy = np.mean(accuracy_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)
    average_auc_roc = np.mean(auc_roc_scores)
    average_specificity = np.mean(specificity_scores)
    average_balance = np.mean(balanced_accuracy_scores)

    avgScores = (average_accuracy, average_precision, average_recall, average_f1, average_auc_roc, average_specificity, average_balance)

    return cm,avgScores, best_scores, best_model, y_test_binarized_best, y_score_best



catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, loss_function='Logloss', verbose=False)
cm, avgScores, best_scores, best_model, y_test_binarized_best, y_score_best = runModel_CatBoost(catboost_classifier, X_encoded, y, n_iterations=1, displayOutput=False)

Scores = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC-ROC', 'Specificity', 'Balanced Accuracy'],
    'Average': [average_accuracy, average_precision, average_recall, average_f1, average_auc_roc, average_specificity, average_balance],
    'Best': [best_accuracy, best_precision, best_recall, best_f1, best_auc_roc, best_specificity, best_balance],
}


bestVisualization = displayAUC(y_test_binarized_best,y_score_best,class_names,title = 'Best model')

"""### Holdout Validation"""

holdout_test = holdout[target]
holdout1 = holdout.drop(target, axis=1)
holdout1 = holdout1[selected_cols]
holdout1 = encoder.fit_transform(holdout1)

y_pred = best_model.predict(holdout1)
y_score = best_model.predict_proba(holdout1)[:, 1]
y_test_binarized = label_binarize(holdout_test, classes=[0, 1])

accuracy = accuracy_score(holdout_test, y_pred)
precision = precision_score(holdout_test, y_pred, average='binary', zero_division=0)
recall = recall_score(holdout_test, y_pred, average='binary',zero_division=0)
f1 = f1_score(holdout_test, y_pred, average='binary')        
specificity = specificity_score(holdout_test,y_pred,average='binary')
balanced_accuracy = (recall + specificity) / 2
auc_roc = roc_auc_score(y_test_binarized, y_score, average='weighted')

Scores['Holdout'] = [accuracy, precision, recall, f1, auc_roc, specificity, balanced_accuracy]

class_names  = ['Yes','No']
holdout_visualization = displayAUC(y_test_binarized,y_score,class_names,title = 'Holdout')

cm2 = confusion_matrix(holdout_test, y_pred)

"""### Display Output"""

scoresDF = pd.DataFrame(Scores)
scoresDF.set_index('Metric', inplace=True)
scoresDF['Difference'] = scoresDF['Best'] - scoresDF['Holdout']
scoresDF['Sign'] = scoresDF['Difference'].apply(lambda x: '+' if x >= 0 else '-')
display(scoresDF)

plot_confusion_matrix(cm,class_names,'Best')
bestVisualization.show()

plot_confusion_matrix(cm2,class_names,'Best')
holdout_visualization.show()