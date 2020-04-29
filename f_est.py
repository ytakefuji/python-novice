import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data=pd.read_csv('f.csv')
x=data['num']
x=sm.add_constant(x)
y=data['f']
est=sm.OLS(y,x).fit()
print(est.params)
e=est.predict(x).astype(int)
x=data['num']
plt.scatter(x,y,c='red')
plt.scatter(x,e,c='blue')
plt.show()
