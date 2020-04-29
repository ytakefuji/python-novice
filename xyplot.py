import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_csv('f.csv')
plt.scatter(data['f'],data['num'])
plt.show()

