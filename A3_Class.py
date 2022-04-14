from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sys, matplotlib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import  PrecisionRecallDisplay, RocCurveDisplay
from matplotlib import pyplot as plt
def main():
    file = sys.argv[1]
    file2 = sys.argv[2]
    df = pd.read_csv(file, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    pd.set_option('display.max_columns', None)

    #print(df)
    data = df.to_numpy() ##this data has a Y
    testData = df2.to_numpy() ##this data does not
    #print(len(data))
    answers = data[ :,143]#[row:row,column:column]
    data = data[ :,0:143] ##returns all but last column.
    #print(data)
    print(answers)
    print(len(testData[0]))
    print(len(answers))
    print(len(data))
    gridcv = 10 ##default behaviour is a stratifiedKFold.

    out = LogisticRegression()
    #out.fit(data,answers) ##only for checking coef's of model, since GridSearchCV doesn't preserve that on its own.
    #print(out.coef_)
    parameters = {}
    cv1 = GridSearchCV(out, param_grid=parameters, scoring = 'precision', cv= gridcv) #Again, not searching anything. Baseline model.
    cv1.fit(data,answers)
    candidate1Score = cv1.best_score_ #included for consistency, not because it's actually useful.
    candidate1Model = cv1.best_params_#see above
    cv1predicts = cv1.predict(data)

    rss = np.sum(np.square(answers - cv1predicts ))
    print("RSS for model 1 is: "+str(rss))
    cv1CvResults = pd.DataFrame.from_dict(cv1.cv_results_)
    print(cv1CvResults.loc[[cv1.best_index_]])

    out = SGDClassifier()
    parameters = {'alpha':(0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5, 10, 50, 100), 'loss':( 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),'penalty':('l2','l1','elasticnet'), 'max_iter':([2000])} ##alphabasically chosen to get a good range plus a test or two to rescue from overfit.
    cv2 = GridSearchCV(out, param_grid=parameters, scoring = 'precision', n_jobs=-1, cv= gridcv)
    cv2.fit(data,answers)
    candidate2Score = cv2.best_score_
    candidate2Model = cv2.best_params_
    cv2CvResults = pd.DataFrame.from_dict(cv2.cv_results_)
    print(cv2CvResults.loc[[cv2.best_index_]])  

    out = KNeighborsClassifier()
    parameters = {'n_neighbors':(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21),'weights':('distance','uniform'),'p':(1, 2)}
    cv3 = GridSearchCV(out, param_grid=parameters, scoring= 'precision', cv = gridcv)
    cv3.fit(data, answers)
    candidate3Score = cv3.best_score_
    candidate3Model = cv3.best_params_
    cv3CvResults = pd.DataFrame.from_dict(cv3.cv_results_)
    print(cv3CvResults.loc[[cv3.best_index_]])

    print("Logistic Regression Best Score: "+str(candidate1Score))
    print("Scored with: "+str(candidate1Model))
    print("SGDClassifier Best Score: "+str(candidate2Score))
    print("Scored with: "+str(candidate2Model))
    print("KNN Best Score: "+str(candidate3Score))
    print("Scored with: "+str(candidate3Model))
    
    
    if((candidate1Score>candidate2Score) & (candidate1Score>candidate3Score)):
        bestModel= candidate1Model
        bestEstimator = cv1.best_estimator_
        print("Best Model is Logistic Regression")
        out = LogisticRegression() ##there are no params.
    elif((candidate2Score>candidate1Score) & (candidate2Score>candidate3Score)):
        bestModel= candidate2Model
        bestEstimator = cv2.best_estimator_
        print("Best Model is Stochastic Gradient Descent Classifier")
        out = SGDClassifier(bestModel['loss'], alpha =['alpha'],penalty=['penalty'],max_iter=['max_iter'].astype(int))
    elif((candidate3Score>=candidate1Score) & (candidate3Score>=candidate2Score)): ##if there's a tie between models, KNeighborsClassifier wins. 
        bestModel= candidate3Model
        bestEstimator = cv3.best_estimator_
        print("Best Model is K-Nearest Neighbors")
        out = KNeighborsClassifier(bestModel['n_neighbors'],weights= bestModel['weights'], p= bestModel['p'])
    else:
        raise Exception("A bug was encountered with determining the best model. ")
    ##code continues now with known best model.
    out.fit(data,answers)
    bestModelPredicts= out.predict(data) ##training data
    rss2 = np.sum(np.square(answers - bestModelPredicts ))
    print("RSS for best model is: "+str(rss2))

    ##Test data predictions/outputs
    bestModelPredictions = out.predict_proba(testData)
    #print(bestModelPredictions)
    #print(bestModelPredictions[1])
    saveMe= 'A3_predictions_group51.txt'
    np.savetxt(saveMe, bestModelPredictions[:,1:2], fmt= '%.2f')
    
    ##Plot onto two graphs.
    PRC = PrecisionRecallDisplay.from_estimator(cv1.best_estimator_, data, answers)
    ROC = RocCurveDisplay.from_estimator(cv1.best_estimator_,data, answers)
    PrecisionRecallDisplay.from_estimator(cv2.best_estimator_, data, answers, ax= PRC.ax_)
    RocCurveDisplay.from_estimator(cv2.best_estimator_,data, answers, ax= ROC.ax_)
    PrecisionRecallDisplay.from_estimator(cv3.best_estimator_, data, answers, ax= PRC.ax_)
    RocCurveDisplay.from_estimator(cv3.best_estimator_,data, answers, ax= ROC.ax_)
    plt.plot([0,1],[0,1], label = "Random Classifier")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
