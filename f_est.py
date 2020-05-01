import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data=pd.read_csv('f.csv')
x=data['num']
x=sm.add_constant(x)
y=data['f']
est=sm.OLS(y,x).fit()
print(est.params)
print(est.rsquared)
print(est.rsquared.astype(float).round(6))
e=est.predict(x).astype(int)
x=data['num']
plt.scatter(x,y,c='red')
#plt.scatter(x,e,c='blue')
plt.scatter(x,x*71-7,c='black')
from sklearn.metrics import r2_score
print(r2_score(y,x*71-7))
plt.show()
