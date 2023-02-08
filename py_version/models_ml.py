import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Machine Learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Model evaluation libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



### Random Forest Classfier
rf = RandomForestClassifier()

### Support Vector Classifier
svc = SVC()

### Logistic Regression 
lr = LogisticRegression(solver='liblinear')

### K Nearest Neighbors
knn = KNeighborsClassifier()

x_data = np.load('featurized_data.npy', allow_pickle = True)
y_data = np.load('labels.npy', allow_pickle = True)

if __name__ == "__main__":
    rf_f_scores = cross_val_score(rf, x_data, y_data, cv=5)
    rf_f_acc = np.mean(rf_f_scores)

    svc_f_scores = cross_val_score(svc, x_data, y_data, cv=5)
    svc_f_acc = np.mean(svc_f_scores)

    lr_f_scores = cross_val_score(lr, x_data, y_data, cv=5)
    lr_f_acc = np.mean(lr_f_scores)

    knn_f_scores = cross_val_score(knn, x_data, y_data, cv=5)
    knn_f_acc = np.mean(knn_f_scores)

    # Visualize performance
    data_r = {'RF':rf_f_acc, 'SVC':svc_f_acc, 'LR':lr_f_acc, 'kNN':knn_f_acc}
    algorithm = list(data_r.keys())
    accuracy = list(data_r.values())
    fig = plt.figure(figsize = (10, 5))
    plt.bar(algorithm, accuracy, color ='red', width = 0.4)
    plt.xlabel("ML models", fontsize = 18)
    plt.ylabel("5 fold accuracy", fontsize = 18)
    plt.title("Result", fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylim([0, 1])
    plt.show()

    print('Random Forest Accuracy: ', rf_f_acc*100)
    print('Support Vector Classifier Accuracy: ', svc_f_acc*100)
    print('Logistic Regression Accuracy: ', lr_f_acc*100)
    print('K Nearest Neighbours Accuracy: ', knn_f_acc*100)


    ### Retraining RF on shuffeled data
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(7):
        current_class_data = x_data[i*20: i*20 + 20]
        X_train.append(current_class_data[0: 16])
        X_test.append(current_class_data[16: ])
        current_class_labels = y_data[i*20: i*20 + 20]
        y_train.append(current_class_labels[0: 16])
        y_test.append(current_class_labels[16: ])
    X_train = np.array(X_train).reshape(-1, 320)
    X_test = np.array(X_test).reshape(-1, 320)
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    print('Accuracy: ', accuracy)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in "0123456"], columns = [i for i in "0123456"])
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    # Dropping class 4 Datapoints
    idx = (y_data != 4)
    x_data = x_data[idx]
    y_data = np.array([i for i in range(6) for j in range(20)])

    #Retrain shallow ML algorithms without class 4
    rf = RandomForestClassifier()
    rf_f_scores = cross_val_score(rf, x_data, y_data, cv=5)
    rf_f_acc = np.mean(rf_f_scores)

    svc = SVC()
    svc_f_scores = cross_val_score(svc, x_data, y_data, cv=5)
    svc_f_acc = np.mean(svc_f_scores)

    lr = LogisticRegression(solver='liblinear')
    lr_f_scores = cross_val_score(lr, x_data, y_data, cv=5)
    lr_f_acc = np.mean(lr_f_scores)

    knn = KNeighborsClassifier()
    knn_f_scores = cross_val_score(knn, x_data, y_data, cv=5)
    knn_f_acc = np.mean(knn_f_scores)
    
    
    data_r = {'RF':rf_f_acc, 'SVC':svc_f_acc, 'LR':lr_f_acc, 'kNN':knn_f_acc}
    algorithm = list(data_r.keys())
    accuracy = list(data_r.values())
    fig = plt.figure(figsize = (10, 5))
    plt.bar(algorithm, accuracy, color ='red', width = 0.4)
    plt.xlabel("ML models", fontsize = 18)
    plt.ylabel("5 fold accuracy", fontsize = 18)
    plt.title("Result", fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylim([0, 1])
    plt.show()
 

    print('Random Forest Accuracy: ', rf_f_acc*100)
    print('Support Vector Classifier Accuracy: ', svc_f_acc*100)
    print('Logistic Regression Accuracy: ', lr_f_acc*100)
    print('K Nearest Neighbours Accuracy: ', knn_f_acc*100)



    # Creating train and test set without class 4
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(6):
        current_class_data = x_data[i*20: i*20 + 20]
        X_train.append(current_class_data[0: 16])
        X_test.append(current_class_data[16: ])
        current_class_labels = y_data[i*20: i*20 + 20]
        y_train.append(current_class_labels[0: 16])
        y_test.append(current_class_labels[16: ])
    X_train = np.array(X_train).reshape(-1, 320)
    X_test = np.array(X_test).reshape(-1, 320)
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    # Training the best model (Random Forest)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    print('Accuracy: ', accuracy)

    # See new confusion matrix of best model without class 4
    conf_matrix = confusion_matrix(y_test, predictions)
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in "012356"], columns = [i for i in "012356"])
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    