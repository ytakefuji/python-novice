# python-novice
<pre>
The goal of "python-novice" is for novice to learn the basic csv file manipulations 
(read, write, update, columns operations) and to practice basic supervised machine 
learning including classification and regression. Integer, floating point, string, 
array, list, and set operations are introduced. Data preprocessing, and data 
augmentation are addressed for machine learning. Graph plotting using matplotlib
is also described.

# [important message to novice]
# If you don't know technical terms, "how to install", "how to run" or "fixing errors", 
# use google search. You should copy and paste a part of error message for fixing errors.
# 
# If you don't know "how to run bash shell on windows", DO google search.
# For Mac users, DO google search using "how to run bash on Mac terminal"
# Remember there is no perfect explanation for understanding something new.
#
# TIPS of google search: 
# strings google search:
# +"yoshiyasu takefuji" where plus '+' includes double-quoted (") phrase.
# In -"yoshiyasu",  minus '-' excludes string "yoshiyasu".
# site search: 
# takefuji site:nature.com
# file type:
# "neural network parallel computing" filetype:pdf
# there are many more tips in google search.
# [end of important message]

There are Python 2.X and 3.X. Python 2.X will be obsolete. 
Therefore, all explanations are based on Python 3.X.
In order to install Python, use miniconda.
https://docs.conda.io/en/latest/miniconda.html

On Windows, download Python3.7 exe file and click it
On Mac, download Python sh file and run 
$ bash the downloaded .sh file on bash shell.
All examples can be practiced on bash shell.

In order to install Python library, run the following command
$ pip install library_name (matplotlib, pandas, ipython,...)
To find the library name,
$ pip search library_name

In order to know python version
$ python -V
To find the location of Python, run the following command
$ which python

# before running your python or ipython, run source command for correcting PATH of 
# python or ipython:
$ source .bashrc

For Python programming:
 -space describes the structure:
 -to practice Python, use ipython (interactive Python)
0. numbers(integer, floating), list, string, set
$ ipython
# concatenate strings
In [1]: a='takefuji'
In [2]: b=' '
In [3]: c='yoshiyasu'
In [4]: a+b+c
Out[4]: 'takefuji yoshiyasu'

# statistics
In [1]:  a=[0,1,2,3,0]
In [2]: import statistics as st
In [3]: st.mean(a)
Out[3]: 1.2
In [4]: st.mode(a)
Out[4]: 0
In [5]: st.stdev(a)
Out[5]: 1.3038404810405297
# float format
In [6]: f'{st.stdev(a):3.2f}'
Out[6]: '1.30'
In [7]: st.variance(a)
Out[7]: 1.7
# isinstance()
In [8]: isinstance(a,list)
Out[8]: True
In [9]: isinstance(a,float)
Out[9]: False

# set function
# set(): to convert any of the iterable to the distinct element 
# and sorted sequence of iterable elements.
In [1]: a = set('abracadabra')
In [2]: a
# unique letters in a
Out[2]: {'a', 'b', 'c', 'd', 'r'}
#
In [3]: basket= ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
In [4]: basket
Out[4]: ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
In [5]: set(basket)
Out[5]: {'apple', 'banana', 'orange', 'pear'}
In [6]: basket.count('apple')
Out[6]: 2

# logical_or each element in two lists
In [1]: a=[0,1,0,0]
In [2]: len(a)
Out[2]: 4
In [3]: b=[1,1,0,1]
In [4]: import numpy as np
In [5]: np.logical_or(a,b).astype(int)
Out[5]: array([1, 1, 0, 1])

# convert numpy array to list using tolist()
# tolist() function returns a list of the values
In [5]: np.logical_or(a,b).astype(int).tolist()
Out[5]: [1, 1, 0, 1]

# append two lists
In [6]: a+b
Out[6]: [0, 1, 0, 0, 1, 1, 0, 1]

# add each element in two lists
In [1]: a=[0,1,0,0]
In [2]: b=[1,1,0,1]
In [3]: from operator import add
In [4]: list(map(add,a,b))
Out[4]: [1, 2, 0, 1]
# map() function returns a map object(which is an iterator) of the results after 
# applying the given function to each item of a given iterable (list, tuple etc.)
# Syntax :
# map(fun, iter)

# add each element in two lists using numpy
In [5]: import numpy as np
In [6]: a=np.array(a)
In [7]: b=np.array(b)
In [8]: a+b
Out[8]: array([1, 2, 0, 1])

# add each element (floating point) in two lists with rounded
In [1]: a=[0.1,1.3,3.6]
In [2]: import numpy as np                                                 
In [3]: np.add(a,a)  
Out[3]: array([0.2, 2.6, 7.2])
In [4]: np.add(a,a).round()                                                
Out[4]: array([0., 3., 7.])

# add (append) each element (string) in two lists using zip
In [5]: a=['take','fuji']
In [6]: [i+j for i,j in zip(a,a)]                                         
Out[6]: ['taketake', 'fujifuji']
In [7]: b=['fuji','san']                                                  
In [8]: [i+j for i,j in zip(a,b)]                                         
Out[8]: ['takefuji', 'fujisan']

# to_string(): to render a string representation from given series object 

</pre>
<pre>
# ASSIGNMENT:
# subtract each element in two lists
</pre>
<pre>
# ASSIGNMENT:
# logical_AND each element in two lists.
</pre>
<pre>
# ASSIGNMENT:
# subtract each element (string) in two lists
</pre>

<pre>
# download ice.csv from 
http://web.sfc.keio.ac.jp/~takefuji/ice.csv
# ice.csv is composed of 4 columns and 31 instances: date,ice,temp,street

# ASSIGNNMENT:
# read ice.csv file and find the largest value in ice. 
# hint: d['ice'].max() shows the max value in ice
# hint: d['ice'].idxmax() shows index of the max value in ice
# hint: d['temp'][d['ice'].idxmax()] shows temp value when ice is the max value.
# show street value when ice is the max value.
#  
# ASSIGNMENT:
# show ice value when street is the max value.
#
# show max value in ice and street.
# hint: dd=d.drop(['date','temp'],axis=1) shows eliminating date and temp columns from d.
# hint: stack() reshapes multiple indexes to single index.
# show index of max value in ice and street.
#
# ASSIGNMENT:
# sort dd based on 'ice' value.
# hint: sort_values('ice')
</pre>

<pre>
1. The goal of this example: 
 -read a csv file (7 columns) and create a csv file co2_v2 (2 columns) 
  with 3 extracted column chunks from the csv file. 
 -Plot a graph with 5th column (y-axis) and x-axis (first column+second column:year_month)

#            decimal     average   interpolated    trend    #days
#             date                             (season corr)
2015  12    2015.958      401.85      401.85      402.51     30
2016   1    2016.042      402.56      402.56      402.27     27
2016   2    2016.125      404.12      404.12      403.31     25
2016   3    2016.208      404.87      404.87      403.39     28
2016   4    2016.292      407.45      407.45      404.63     25
2016   5    2016.375      407.72      407.72      404.27     29
2016   6    2016.458      406.83      406.83      404.49     26
2016   7    2016.542      404.41      404.41      404.07     28
2016   8    2016.625      402.27      402.27      404.18     23
...
</pre>
<pre>
# read csv file co2_v2.txt which should be in the same folder
# download cov2_v2.txt file from 
# https://raw.githubusercontent.com/ytakefuji/global-warming/master/co2_v2.txt
# run the following command in ipython
co2=open('co2_v2.txt','r',encoding='utf-8')
# comment by #
'''
comments by three single quotes (''')
comments
'''
# create an empty list called data 
data=[]
# or data=list()
# in order to extract 3 columns (year, month, co2) and create 2 columns (year_month, co2)
# for loop, list string append 
# a single space indicates internal structure of for loop
# remember co2_v2.txt has 7 chunks: each chunk will be stored in variables (a,b,c,d,e,f,g)
# concatenate strings by using the + operator
# single quote (') or double quote (") makes a string: '_' and ','
# split() separates chunks of a line of co2_v2.txt
for i in co2:
 a,b,c,d,e,f,g=i.split()
 data.append(a+'_'+b+','+e)
# open and write file co2_v2 using 'w' write mode
f=open('co2_v2','w',encoding='utf-8')
f.write("\n".join(data))
f.close()
</pre>
<pre>
# co2_v2: 2 columns data
2015_12,401.85
2016_1,402.56
2016_2,404.12
2016_3,404.87
2016_4,407.45
2016_5,407.72
2016_6,406.83
2016_7,404.41
2016_8,402.27
...
</pre>
<pre>
# ASSIGNMENT:
# print co2 all data only
</pre>

<pre>
# import matplotlib library for graph
import matplotlib.pyplot as plt
# import pandas library for column manipulations
import pandas as pd
data=pd.read_csv('co2_v2')
# attach two columns names to data: year and co2
data.columns=['year','co2']
# x-axis name
plt.xlabel('year')
# y-axis name
plt.ylabel('density')
# x-axis ticks which should be vertical
plt.xticks(rotation='vertical',fontsize=6)
# plot dot black colored graph: x and y
plt.plot(data['year'],data['co2'],'k.')
fig=plt.figure(1)
# size of graph
fig.set_size_inches(10,5)
# save co2_v2.png file
plt.savefig('co2_v2.png',dpi=fig.dpi,bbox_inches='tight')
plt.show()
plt.close()
</pre>
<pre>
co2_v2.png image file should be created.
</pre>
<img src="https://github.com/ytakefuji/global-warming/blob/master/co2_v2.png" height=200 width=400>
<pre>
You should download the weekly dataset from the following site:
ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt
and extract the data of the last two years to create a file named co2w.txt.
<br>
ASSIGNMENT:
You should plot a graph with x-axis(year_month_day) and y-axis(co2) using co2w.txt
<br>
ASSIGNMENT: 
Build the same program based on Python2.X.

</pre>

<pre>
# There are regression and classification in supervised machine learning.
# Machine Learning: Random Forest Regression (floating point)
# download ice.csv from:
# https://raw.githubusercontent.com/ytakefuji/ensemble-machine-learning/master/ice.csv
# ice.csv: ice(ice sales),temp(highest temperature),street(no. of pedestrians)
date,ice,temp,street
2012/8/1,12220,26,4540
2012/8/2,15330,32,5250
2012/8/3,11680,32,6000
2012/8/4,12640,29,5120
2012/8/5,15150,34,4640
2012/8/6,16440,33,8620
2012/8/7,16080,35,5810
2012/8/8,9830,34,4170
...

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# read ice.csv file
data=pd.read_csv('ice.csv')
# y=f(x)=f(temp,street) where y is ice sales which we would like to predict
# x is temp and street
x=data[['temp','street']]
y=data['ice']
# RandomForestRegressor: n_estimators is no. of trees
# min_samples_split specifies the minimum number of samples required 
# to split an internal leaf node.
clf=RandomForestRegressor(n_estimators=50, min_samples_split=2)
# Machine Learning
clf.fit(x,y)
# accuracy score
print(clf.score(x,y))
# feature_importances_
print(clf.feature_importances_)
# predict p=f(x)
p=clf.predict(x)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--b')
plt.plot(t,p,'-b')
plt.legend(('real','randomF'))
plt.show()
</pre>
<pre>

# Machine Learning: Random Forest Classification (integer)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
data=pd.read_csv('ice.csv')
x=data[['temp','street']]
y=data['ice']
clf=RandomForestClassifier(n_estimators=82, min_samples_split=2)
clf.fit(x,y)
print(clf.score(x,y))
print(clf.feature_importances_)
p=clf.predict(x)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--b')
plt.plot(t,p,'-b')
plt.legend(('real','randomF'))
plt.show()
</pre>

<pre>
# ASSIGNMENT: random forest using ice.csv
# build regression program and classification programs using ice.csv respectively.
# Use train_test_split function which is described in:
# https://github.com/ytakefuji/titanic
# where test_size=0.2 should be used in your program.
</pre>

<pre>
# ASSIGNMENT: binary classification using random forest classification.
# read titanic folder and develope a binary classification program
# https://github.com/ytakefuji/titanic
# use train_test_split function.
# show what are importances in the features.
# You must understand preprocessing and train_test_split.
# build a program using dataset of machine-learning-in-medicine/pima-indians-diabetes.csv 
# pima csv (9 columns)
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
...
# parameter names
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
# feature_importances
dic=dict(zip(X.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))

</pre>

<pre>
# ASSIGNMENT: binary classification with SMOTE
# use SMOTE for imbalanced data in pima-indians-diabetes.csv.
# develope a binary classification with random forest classification.
</pre>
