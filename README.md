# python-novice
<pre>
There are Python 2.X and 3.X. Python 2.X will be obsolete. 
Therefore, all explanations are based on Python 3.X.
In order to install Python, use miniconda.
On Windows, download Python3.7 exe file and click it
On Mac, download Python sh file and run 
$ bash the downloaded .sh file 

In order to install Python library, run the following command
$ pip install library_name (matplotlib, pandas, ipython,...)
To find the library name,
$ pip search library_name

In order to know python version
$ python -V
To find the location of Python, run the following command
$ which python

For Python programming:
 -space describes the structure:
 -to practice Python, use ipython (interactive Python)
 
1. The goal of this example: 
 -read a csv file (7 columns) and create a csv file co2_v2 (2 columns) 
  with 3 extracted column chunks from the csv file. 
 -Plot a graph with 5th column (y-axis) and x-axis (first column+second column:year_month)

2015  12    2015.958      401.85      401.85      402.51     30
2016   1    2016.042      402.56      402.56      402.27     27
2016   2    2016.125      404.12      404.12      403.31     25
2016   3    2016.208      404.87      404.87      403.39     28
2016   4    2016.292      407.45      407.45      404.63     25
2016   5    2016.375      407.72      407.72      404.27     29
2016   6    2016.458      406.83      406.83      404.49     26
2016   7    2016.542      404.41      404.41      404.07     28
2016   8    2016.625      402.27      402.27      404.18     23
</pre>
<pre>
# how to read csv file co2_v2.txt which should be in the same folder
# in order to check the existence of co2_v2.txt, run ls command
co2=open('co2_v2.txt','r',encoding='utf-8')
# comment
"""
comments
comments
"""
# create an empty list called data
data=[]
# or data=list()
# in order to extract 3 columns (year, month, co2) and create 2 columns (year_month, co2)
# for loop, list append
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
