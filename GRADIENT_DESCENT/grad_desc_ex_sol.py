import pandas as pd
import numpy as np
from sklearn import linear_model
import math 

def gradient_dec(x,y):
    m_curr = b_curr = 0 
    iterns = 1000000
    cost_prev = 0
    learnin_rate = 0.0002
    n = len(x)

    for i in range(iterns):
        y_pred = m_curr*x + b_curr 
        m_var = -(2/n)*sum((y-y_pred)*x)
        b_var = -(2/n)*sum(y-y_pred)
        cost = (1/n)*sum((y-y_pred)**2)
        m_curr = m_curr - learnin_rate*m_var 
        b_curr = b_curr - learnin_rate*b_var 

        if(math.isclose(cost,cost_prev,rel_tol=1e-20)):
            break 
        cost_prev = cost 
            
    print(m_var)
    print(b_var)
    print("m_grad_descent {}| b_grad_descent {}| cost {}".format(m_curr,b_curr,cost))


df = pd.read_csv("grad_desc_ex.csv")
#print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['math']],df.cs)

for i in reg.coef_:
    m_lin_reg = i

b_linear_reg = reg.intercept_

print("m_lin_reg {}| b_lin_reg {}".format(m_lin_reg,b_linear_reg))


x = np.array(df.math)
y = np.array(df.cs)

gradient_dec(x,y)