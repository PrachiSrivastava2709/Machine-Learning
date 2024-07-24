from pandas import read_csv
from math import isclose
from sklearn.linear_model import LinearRegression

def gradient_descent(x, y):
    '''
    Given x & y values, performs gradient descent algorithm & 
        finds best fit line (slope and intercept) {Linear Regression}
    '''
    n = len(x)
    iterations = 1000000
    learning_rate = 0.0002
    m_curr = c_curr = 0 #initial values to slope and intercept
    prev_cost = 0
    for i in range(iterations):
        y_predicted = (m_curr * x) + c_curr #mx + c
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted)) #partial derivative of cost(mse) wrt m(slope)
        cd = -(2 / n) * sum(y - y_predicted) #partial derivative of cost(mse) wrt c(intercept)
        m_curr = m_curr - learning_rate * md
        c_curr = c_curr - learning_rate * cd
        if isclose(prev_cost, cost, rel_tol = 1e-20):
            break
        prev_cost = cost

    print(f'cost = {cost}, m = {m_curr}, c = {c_curr}')
    return

def sk_regression(x, y):
    '''
    Given x & y values, implements linear regression from sci kit learn
        This is to compare values obtained by our GD algorithm and via sci kit learn
    '''
    model = LinearRegression().fit(x, y)
    print(f'Coefficients: {model.coef_}, Intercept: {model.intercept_}')
    return


if __name__ == "__main__":
    data = read_csv("gradient_descent\scores.csv") #returns a data frame object
    math_scores = data['math'].to_numpy() #returns a numpy array
    cs_scores = data['cs'].to_numpy()

    gradient_descent(math_scores, cs_scores)
    math = data.drop(['name', 'cs'], axis = 1) # doing this to match the datatype to pass in the fit method
    sk_regression(math, cs_scores)
