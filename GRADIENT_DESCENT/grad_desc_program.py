import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0 
    iterations = 1000 

    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr 
        #print(y_predicted)

        cost = (1/n)*sum((y-y_predicted)**2)
        
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - (md)*(learning_rate)
        b_curr = b_curr - (bd)*(learning_rate)
        
        print("m {}|b {}|iteration {}|cost {}".format(m_curr,b_curr,i,cost))


x=np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)