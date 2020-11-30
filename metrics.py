from sklearn.metrics import explained_variance_score
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Function to print the Receiver Operating Characteristics curve, and calculate AuC
def auc_roc(y_test, test, train, classifier):
    pred_probs = classifier.predict_proba(test)
    train_probs = classifier.predict_proba(train)[:,1]
    noSkillProb = [0 for _ in range(len(y_test))]
    lr_probs = pred_probs[:,1]

    noSkillAUC = roc_auc_score(y_test,noSkillProb)
    logRegAUC = roc_auc_score(y_test,lr_probs)

    print('No Skill: ROC AUC=%.3f' % (noSkillAUC))
    print('Logistic: ROC AUC=%.3f' % (logRegAUC))

    #FPR,TPR,thresholds = roc_curve(y_test,)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, noSkillProb)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    t_fpr,t_tpr,_ = roc_curve(y_train,train_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.plot(t_fpr,t_tpr,marker='*',label='Training')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

# Function to calculate performance metrics of the model
def model_metrics(classifier,y_test,pred,score, only_acc=0):
    print("Accuracy:", score*100, "%")
    if only_acc == 0:
        print("Precision:",precision_score(y_test,pred)*100,"%")
        print("Recall:",recall_score(y_test,pred)*100,"%")
        print("F1 Score:",f1_score(y_test,pred)*100,"%")
        print("MSE:",mean_squared_error(y_test,pred)*100,"%")
        print("Explained Variance Regression Score:", explained_variance_score(y_test,pred))
    # auc_roc(y_test, classifier)

def kfold_cross_validate(LR, Xtf):
    kfold = KFold(n_splits=10,shuffle=True)
    # LR = LogisticRegression()
    scores = cross_val_score(LR,Xtf,y,cv=kfold, scoring='accuracy', n_jobs=-1)
    mean_acc = np.mean(scores)*100
    std_acc = np.std(scores)*100
    print("Mean Accuracy: %0.2f"%mean_acc, "%")
    print("Standard Deviation of Accuracy: %0.2f"%std_acc,"%")