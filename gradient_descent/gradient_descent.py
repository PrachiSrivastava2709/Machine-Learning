import numpy as np

def gradient_descent(x, y):
    m_curr = c_curr = 0
    iterations = 10000 
    learning_rate = 0.08 #should be as maximum as possible (trial & error - hyperparameter tuning)
    n = len(x)

    for i in range(iterations):
        y_predicted = m_curr * x + c_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted)) #partial derivative of cost(mse) wrt m(slope)
        cd = -(2 / n) * sum(y - y_predicted) #partial derivative of cost(mse) wrt c(intercept)
        m_curr = m_curr - learning_rate * md
        c_curr = c_curr - learning_rate * cd
        print ("m {}, c {}, cost {} iteration {}".format(m_curr, c_curr, cost, i))

    return

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x, y)