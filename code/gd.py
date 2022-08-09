import numpy as np
import matplotlib.pyplot as plt
from svm_plot import plot_decision_function 

import IPython

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOs. Fill the cost function, the gradient function and gradient descent solver.
"""

def ex_4_a(x, y):
    
    # TODO: Split x, y (take 80% of x, and corresponding y). You can simply use indexing, since the dataset is already shuffled.
    n_split = round(0.8*x.shape[0])
    X_train = x[0:n_split,:]
    y_train = y[0:n_split]
    
    #Set penalty term C to 1
    C = 1
    
    # Define the functions of the parameter we want to optimize
    f = lambda th: cost(th, X_train, y_train, C)
    df = lambda th: grad(th, X_train, y_train, C)
    
    # TODO: Initialize w and b to zeros. What is the dimensionality of w?
    
    w = np.zeros((X_train.shape[1],1))
    b = 0
    
    eta = 0.08
    max_iter = 20
    
    
    theta_opt, E_list = gradient_descent(f, df, (w, b), eta, max_iter)
    w, b = theta_opt
    
    
    
    # TODO: Calculate the predictions using the test set
    X_test = x[n_split:]
    y_test = y[n_split:]
    m = y_test.size
    y_calc = - np.ones((m,1))
    bool_y_calc = np.dot(w.T, X_test.T) + b >= 0
    y_calc[bool_y_calc.T] = 1
    
    
    # TODO: Calculate the accuracy    
    y_error = y_calc.T != y_test
    
    acc = 1 - np.sum(y_error)/m

    
    #print results
    print("w_opt = \n", w , " \n \nb_opt =", b)
    print("cost func = ", f((w,b)))
    
    print("accuracy = ", acc*100, "%")

    

    
    # Plot the list of errors
    if len(E_list) > 0:
        fig, ax = plt.subplots(1)
        ax.plot(E_list, linewidth=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Error')
        ax.set_title('Error monitoring')
        
    # TODO: Call the function for plotting (plot_decision_function).
    
    plot_decision_function((w,b), X_train, X_test, y_train, y_test)
    

def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Finds the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decreases the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm
    
 

    E_list = np.zeros(max_iter)
    w, b = theta0
    
    for counter in range(max_iter):
    
        grad_w, grad_b = df((w, b))
        w = w - learning_rate * grad_w #minus, because the gradient points to the steepest ascend and we want to go down the hill
        b = b - learning_rate * grad_b
        E_list[counter] = f((w, b))
        



    # END TODO
    ###########

    return (w, b), E_list


def cost(theta, x, y, C):
    """
    Computes the cost of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: cost
    """

    w, b = theta
    
    m = y.size
    zero_array = np.zeros((1,m))

    max_term = 1 - np.multiply(y,np.dot(w.T,x.T) + b)

    sum_term = np.sum(np.maximum(zero_array, max_term))
    
    cost = 0.5*np.linalg.norm(w)**2 + C/m * sum_term

    return cost


def grad(theta, x, y, C):
    """

    Computes the gradient of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: grad_w, grad_b
    """
    w, b = theta
    
    m = y.size

    
    II = np.ones((1,m))
    max_term = 1 - np.multiply(y,np.dot(w.T,x.T) + b)
    II[max_term <= 0] = 0
    
    mid_term = np.multiply(II,y)
    sum_term = (x.T * mid_term).T #row wise multiplication    

    grad_w = w.T - C/m*np.sum(sum_term,axis=0)  # TODO 
    grad_w = grad_w.T
    
    
    grad_b = - C/m*np.sum(mid_term)


    
    return grad_w, grad_b
    
    
if __name__ == "__main__":
    import svm_main
    svm_main.main()
    
