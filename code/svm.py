import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from IPython.core.debugger import set_trace
import IPython

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOS are contained here.
"""


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########

    clf = svm.SVC(kernel="linear")
    clf.fit(x, y)
    plot_svm_decision_boundary(clf, x,y)



def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    x = np.vstack((x,[4,0]))
    y = np.hstack((y, 1))

    clf = svm.SVC(kernel="linear")
    clf.fit(x, y)
    plot_svm_decision_boundary(clf, x,y)




def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]
    x = np.vstack((x,[4,0]))
    y = np.hstack((y, 1))


    for c in Cs:
        clf = svm.SVC(kernel="linear", C = c)
        clf.fit(x, y)
        plot_svm_decision_boundary(clf, x,y)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train,y_train,x_test,y_test)
    print("Calculated test score for linear kernel: " ,clf.score(x_test,y_test))
    print("Calculated train score for linear kernel: " ,clf.score(x_train,y_train))


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the training and test scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    trainScore = []
    testScore = []

    degrees = range(1, 21)
    for deg in degrees:
        clf = svm.SVC(kernel="poly", degree = deg, coef0 = 1)
        clf.fit(x_train, y_train)
        testScore.append(clf.score(x_test,y_test))
        trainScore.append(clf.score(x_train,y_train))

    plot_score_vs_degree(trainScore,testScore,degrees)
    clf = svm.SVC(kernel="poly", degree = degrees[testScore.index(max(testScore))], coef0 = 1)
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train,y_train,x_test,y_test)

def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    trainScore = []
    testScore = []

    for gam in gammas:
        clf = svm.SVC(kernel="rbf", gamma = gam)
        clf.fit(x_train, y_train)
        testScore.append(clf.score(x_test,y_test))
        trainScore.append(clf.score(x_train,y_train))

    plot_score_vs_gamma(trainScore,testScore,gammas)

    clf = svm.SVC(kernel="rbf", gamma = gammas[testScore.index(max(testScore))])
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train,y_train,x_test,y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Note that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function (parameter baseline)
    ###########
    
    ## Notes from Mail
    ##... specify the scores obtained with a linear kernel using
    ##the optional argument lin_score_train AND LIN_SCORE_TEST.
    ##Note that the chance level has changed for this example.        
    ##1) Lin_score_test was missing.
    ##2) Regarding the chance level: The hint in the code is not
    ##correct. A linear kernel performs well on images. Please ignore that comment. 
    #Set the parameter baseline to 0.2.
    ##Why? There are 5 possible outcomes (5 possible classes). If the outcome is random and equally probable, the probability of the outcome (the chance level) is then 1/5 = 0.2.

    
    #clfRBF = svm.SVC(kernel="rbf", decision_function_shape='ovr')
    gammas = [10**-5, 10**-4, 10**-3,10**-2, 10**-1, 1, 10, 10**2, 10**3, 10**4, 10**5]
    trainScore = []
    testScore = []
    for gam in gammas:
        clfRBF = svm.SVC(C=10, kernel="rbf",gamma=gam, decision_function_shape='ovr')
        clfRBF.fit(x_train, y_train)
        testScore.append(clfRBF.score(x_test,y_test))
        trainScore.append(clfRBF.score(x_train,y_train))
    
    clfLin = svm.SVC(C=10, kernel="linear", decision_function_shape='ovr')
    clfLin.fit(x_train, y_train)
    plot_score_vs_gamma(trainScore,testScore,gammas,lin_score_train=clfLin.score(x_train,y_train), lin_score_test=clfLin.score(x_test,y_test), baseline=.2)


    
    

def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 images classified as the most misclassified digit using plot_mnist.
    ###########
    clf = svm.SVC(C=10, kernel="linear", decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    
    ConfM=confusion_matrix(y_test, clf.predict(x_test))
    
    #The Confusion matix looks as follows:
    #147    1   2   0   0
    #3      142 0   4   1
    #4      2   126 0   18
    #3      5   0   142 0
    #2      3   9   0   136
    #This means that the middle object is the most wrong classified
    labels = range(1, 6)
    
    
    plot_confusion_matrix(ConfM, labels)
    i = 3  # CHANGE ME! Should be the label number corresponding the largest classification error.
    #sel_err = np.array([0])#np version
    #Default: np.array([0]) CHANGE ME! Numpy indices to select all images that are misclassified.
    sel_err=[]#List version 
    #Tries both versions doenst matter 
    firstTenMiss=0
    
    y_pred=clf.predict(x_test)
    for elemNr in range(0,x_test.shape[0]):
        #if(y_test[elemNr]!=clf.predict(np.reshape(x_test[elemNr], (1,-1)))):
        if(y_test[elemNr]!=y_pred[elemNr]):
            #set_trace()
            if y_pred[elemNr]==i:
                firstTenMiss+=1
                print(y_test[elemNr])
                sel_err.append(elemNr)#listversion
                #sel_err=np.append(sel_err,elemNr)#the numpy Version
                if firstTenMiss==10:
                    break
      
    
    print(sel_err)
    sel_err=np.array(sel_err)#the List version converted to numpy 
    print(sel_err)    
    #set_trace()
    #y_pred=np.array(y_pred).reshape(-1,1)
    
    #Tested the shapes: 
    #With changes: x[sel]=(10,784); y[sel]=(10,1)
    #otherwise (10,) and so on 
    #x[307] for Example gives 784 values where most are 0 and some are between 0 and 1
    
    # Plot with mnist plot

    plot_mnist(x_test[sel_err], y_pred[sel_err],labels=i, k_plots=10, prefix='Predicted class')
    
if __name__=="__main__":
    import svm_main
    svm_main.main()
