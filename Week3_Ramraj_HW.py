import numpy as np
import pandas as pd

#Question 1 Numpy
array_1 = np.array([1, 2, 3])
array_2 = np.array([4, 2, 6])
stack_1 = np.vstack([array_1, array_2])
stack_2 = np.hstack([array_1, array_2])
print(stack_1)
print(stack_2)

#Question 2 Numpy
common_1 = np.intersect1d(array_1, array_2)
print(common_1)

#Question 3 Numpy
range_1 = array_1[np.where((array_1 >= 2) & (array_1 <= 3))]
print(range_1)

#Question 4 Numpy
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
filtered_data = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]
print(filtered_data)

#Question 1 Pandas
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
filter_1 = df.loc[::20, ("Manufacturer", 'Model','Type')]
#df.info()
print(filter_1)
#print(df)

#Question 2 Pandas
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df['Min.Price'].fillna(df['Min.Price'].mean(), inplace=True)
df['Max.Price'].fillna(df['Max.Price'].mean(), inplace=True)

#Question 3 Pandas
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
sum_1 = df[df.sum(axis=1) > 100]
print(sum_1)