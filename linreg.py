import pandas
import numpy as np
from ggplot import *

def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features 
    (input data points) and values (output data points).
    """
    m = len(values)
    predicted_values = np.dot(features, theta) 
    residuals = values - predicted_values
    sum_of_square_residuals = np.square(residuals).sum()
    cost = sum_of_square_residuals / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    This can be the same gradient descent code as in the lesson #3 exercises,
    but feel free to implement your own.
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        predicted_values = np.dot(features, theta) 
        residuals = values - predicted_values

        # record cost
        sum_of_square_residuals = np.square(residuals).sum()
        cost = sum_of_square_residuals / (2*m)
        cost_history.append(cost)

        # update theta
        theta = theta + (alpha/m) * np.dot(residuals, features)

    return theta, pandas.Series(cost_history)

def plot_cost_history(alpha, cost_history):
   """This function is for viewing the plot of your cost history.
   You can run it by uncommenting this

       plot_cost_history(alpha, cost_history) 

   call in predictions.
   
   If you want to run this locally, you should print the return value
   from this function.
   """
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and
    # 'predictions,' returns the coefficient of determination, R^2,
    # for the model that produced predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE

    m = len(data)
    sum_of_square_residuals = np.square(data - predictions).sum()
    variance = np.square(np.mean(data) - data).sum()
    r_squared = 1 - sum_of_square_residuals / variance
    
    return r_squared
