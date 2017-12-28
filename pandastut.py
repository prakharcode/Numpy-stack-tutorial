'''
Pandas is the library for data science in python
pandas is an open source, BSD-licensed library providing
high-performance, easy-to-use data structures and data
analysis tools for the Python programming language.

Has two basic Data structures at its core.
DataFrame
Series

DataFrame is a container for Series, and Series is a container
for scalars and we would like to be able to insert and remove
objects from these containers in a dictionary-like fashion.

Why Pandas instead of numpy?
we would like sensible default behaviors for the common API
functions which take into account the typical orientation
of time series and cross-sectional data sets. When using
ndarrays to store 2- and 3-dimensional data, a burden is placed
on the user to consider the orientation of the data set when writing
functions; axes are considered more or less equivalent (except when
C- or Fortran-contiguousness matters for performance). In pandas, the
axes are intended to lend more semantic meaning to the data; i.e., for
a particular data set there is likely to be a “right” way to orient the data.
The goal, then, is to reduce the amount of mental effort required to code up data
transformations in downstream functions.
For example, with tabular data (DataFrame) it is more semantically helpful to think
 of the index (the rows) and the columns rather than axis 0 and axis 1
'''

import pandas as pd
import numpy as np

#axis 0 is used to refer column-wise iteration
#and 1 for row-wise


# +------------+---------+--------+
# |            |  A      |  B     |
# +------------+---------+---------
# |      0     | 0.626386| 1.52325|----axis=1----->
# +------------+---------+--------+
#                 |         |
#                 | axis=0  |
#                 ↓         ↓


#Two types of data structure in pandas
#1. Series (1-D)
#2. DataFrame (2-D)
#Series is very similar to ndarrays and
#dictionary

#DataFrame is like a spreadsheet or SQL table,
#or a dict of Series objects


#Object Creation || READING

#to call a series data use s= pd.Series('file')
s = pd.Series([1,3,5,np.nan,6,8])
print s

#Object Creation with help numpy11

dates = pd.date_range('20170101', period=6)
print type(dates)

df = pd.DataFrame(np.random.rand(6,4), index=dates, columns=list('ABCD'))

#Creating a dataframe by passing dictionary of series

d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print df.head()


#Creating a dataframe by passing dictionary

df2 = pd.DataFrame({
                    'A':1.,
                    'B':pd.Timestamp('20171225'),
                    'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D':np.array([3]*4, dtype='int32'),
                    'E':pd.Categorical(["test","train","test","train"]),
                    'F':'foo'})

#more on it in docs (page 494)

#Reading from external files

# FormatType | DataDescription   |   Reader       |   Writer     |
# ================================================================
# text       |       CSV         |  read_csv      |  to_csv      |
# text       |      JSON         |  read_json     |  to_json     |
# text       |      HTML         |  read_html     |  to_html     |
# text       |  Localclipboard   |  read_clipboard|  to_clipboard|
# binary     |    MS Excel       |  read_excel    |  to_excel    |
# binary     |  HDF5 Format      |  read_hdf      |  to_hdf      |
# binary     |  FeatherFormat    |  read_feather  |  to_feather  |
# binary     |  Parquet Format   |  read_parquet  |  to_parquet  |
# binary     |     Msgpack       |  read_msgpack  |  to_msgpack  |
# binary     |     Stata         |  read_stata    |  to_stata    |
# binary     |      SAS          |  read_sas      |              |
# binary     |  Python Pickle    |  read_pickle   |  to_pickle   |
# SQL        |      SQL          |  read_sql      |  to_sql      |
# SQL        |  Google Big Query |  read_gbq      |  to_gbq      |

df = pd.read_csv('file', header=0 )
#header[int or list of ints] is the col name by default it is one but can be set to any line number

#names:[array like] is another argument to pass the name of your choice for cols

# index_col:[int or list or False, default none]: Cols to use as row labels of DataFrame.
#If a sequence is given multi-indexing is followed

#usecols [array-like or callable <lambda>]: only used the column specified in this list

#skipinitialspace [boolean, default False] Skip spaces after delimiter
#use skiprows in argument to specify the no of rows to be skipped

# na_values [scalar, str, list-like, or dict, default None] Additional strings to recognize
# as NA/NaN. If dict passed, specific per-column NA values. See na values const below for a
# list of the values interpreted as NaN by default.

# keep_default_na [boolean, default True] If na_values are specified and keep_default_na is False
# the default NaN values are overridden, otherwise they’re appended.

#pd.read_csv(StringIO(data), comment='#')
# use comment argument to parse comment

#use the usecols parameter to eliminate extraneous column data
#that appear in some lines but not others
#pd.read_csv(StringIO(data), usecols=[0, 1, 2])

# sometimes pandas gets confused with the index and puts the first column as index
#to remove the first column as index just put index_col=false, this will move all the
#column one place to right
#MORE ON COLUMN HEADS AND INDEXED IN DOCS (page 1027 onwards)


#the column name is the key value of dictionary
#the values in the row comes from the value list in dictionary
#if the value is not list like the same value is repeated over
#the column
#pd.Timestamp is datetime object in pandas
#Series is a 1-D data type
#Note we can specify index in any column



#READING HTML CONTENT
#The top-level read_html() function can accept an HTML string/file/URL
#and will parse HTML tables into list of pandas DataFrames.

# url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'
# df = pd.read_html(url)

# DataFrame objects have an instance method to_html which renders the
# contents of the DataFrame as an HTML table
df = pd.DataFrame(randn(2, 2))
print(df.to_html())
#it has argument like bold_rows and classes which corresponds to HTML


#READIND EXCEL contents
#read_excel('path_to_file.xls', sheet_name='Sheet1')

# To facilitate working with multiple sheets from the same file, the ExcelFile
# class can be used to wrap the file and can be be passed into read_excel There
# will be a performance benefit for reading multiple sheets as the file is read
# into memory only once

# xlsx = pd.ExcelFile('path_to_file.xls')
# df = pd.read_excel(xlsx, 'Sheet1')


#USING YOUR CLIPBOARD
#A PLACE IN MEMORY WHERE YOUR COPIED ITEMS ARE TEMPORARILY SAVED
# A handy way to grab data is to use the read_clipboard method, which
# takes the contents of the clipboard buffer and passes them to the
# read_table method.

#READING FROM SQL
#pandas.read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True,
# parse_dates=None, columns=None, chunksize=None)
# Read SQL database table into a DataFrame.
#one can also execute sql queries by read_sql()


#IN ADDITION TO THESE METHODS PANDAS OFFERS BUILT-IN APT SUPPORT TO DIRECTLY FETCH
#REAL TIME DATA FROM VARIOUS SITES USING pandas-datareader library
# read more about it here https://pandas-datareader.readthedocs.io/en/latest/


##for better displaying the df, we can use setting mentioned on page 599

print df.shape

print df2.dtypes
# gives information of data types in the respective
#columns
print df2.describe()
#this gives the whole information about the DataFrame
print df2.index
#this gives an array having its element from the
#index column of dataframe
print df2.columns
#prints the column name in dataframe
print df2.values
#prints the values in the dataframe, not index and columns
print len(df)
#gives number of rows
#series.nunique() returns the number of unique non-NA values



#TO VIEW data
print df2 #gives the dataframe
print df2.head(3) #gives the first N rows specified in argument
print df2.tail(3) #gives last N rows specified in the argument


#Sorting a df
df2.sort_index(axis=1,ascending=False)
#returns a sorted data frame row-wise, on column name

df2.sort_values(by='B')
#sorting the dataframe based on the column name given as
#argument

###############ACCESSING DATAFRAME############################
# The basics of indexing are as follows:
#
#    Operation               Syntax        Result
# Select column            df[col]         Series
# Select row by label      df.loc[label]   Series
# Select row by integer    df.iloc[loc]    Series
# location
# Slice rows                df[5:10]       DataFrame
# Select rows by boolean    df[bool_vec]   DataFrame
# vector


#to get complete column values as pandas series
df2['A']
#returns series of the values in column A
#this is dictionary-like access
#Since it is like a dictionary-like access
#we can call it with column names
df.A
#obviously this won't work with number indexes


#Selecting multiple cols
df2[['A','B']]

df2[0:3]
#returns row number 0,1,2 from dataframe
#this is a list-like access, slicing

#this is a matter of great confusion as
#pandas provide both kind of access together
#hence df[0] looks for key '0' not index 0
#and df[0:] looks index number 0 to all

#To access row-wise/ Index-wise pandas has givrn
#functions like loc[], iloc[], ix[]

df2.iloc[1] #accepts index and boolean array of row length only
df2.loc[dates[2]] #accepts label and boolean array of row length only

#both the function accepts the list of respective data, iloc -> list
#of indexes and loc -> list of index labels
df2.iloc[0:2] == df2[0:2]
#this gives element-wise boolean table

#one can access row and column with loc

df2.loc[dates[1:4],['A','B']] #both should be list if more than one query
df2.loc[dates[1],'A'] # for single query
#one can access table like matrix too
df2[1:2]['A']
#note the first should only be list or it starts looking for cols

#With Series, the syntax works exactly as with an ndarray, returning a
#slice of the values and the corresponding labels


# .loc, .iloc, and also [] indexing can accept a callable as indexer.
# The callable must be a function with one argument  and that returns
#valid output for indexing.

df2.loc[lambda df2: df2.A >0]
#similarly
df2[lambda df: df.A >0]

#.ix method is now deprecated hence we'll not discuss that


#to select random row from the dataframe use sample method
df2.sample(4) #pass the no of samples as argument
#or we can ask for a fraction of random dataframe
df2.sample(frac=0.4)


#Since indexing with [] must handle a lot of cases (single-label
# access, slicing, boolean indexing, etc.), it has a bit of overhead
# in order to figure out what you’re asking for. If you only want to
# access a scalar value, the fastest way is to use the at and iat methods,
# which are implemented on all of the data structures.

df2.iat[2,2] #matrix like access and only INDEX
#analogus tp
df2.iloc[2]['C']

df2.at[dates[2],'C'] #matrix like access but only LABELS are used
# analogus to
df2.loc[dates[2],'C']

#indexing using isin

df2[df2.isin(np.random.rand(4))]

#similar to df2[df2.A>0] is
df2.where(df2.A>0) #to make it more readable

#remember df2[df2.A>0] gives the complete dataframe where A>0
# but df[df>0] will give only those val in each col > 0 else NAN

#where is useful as it can use a callable and another function
# callable finds the value other changes it.

df2.where(lambda x: x>4, lambda x: x+10)

# MASKING
# mask() method is used to turn any specific value NAN
df2.mask(df2>0)

#REMOVE DUPLICATE DATA
#using duplicated we can get bool series of duplicate rows
df2.duplicated()
#use drop_duplicate to remove duplicate data
df2.drop_duplicate(keep='first')
#first here means to keep first occurence and remove everything else
#similarly last can be used and False can be given to not keep any


#REINDEXING header column or index row, one can use reindex, it accepts
#axis argument axis='index' or axis='columns' default axis is index.
#df.reindex(['three', 'two', 'one'], axis='columns')

#reindexing to align a dataframe df1 to another dataframe df2 one can use
#df.reindex_like(df2)

#to RENAME a column or index use df.rename()


#As a convenience, there is a new function on DataFrame called reset_index which
#transfers the index values into the DataFrame’s columns and sets a simple integer index

###############ITERATION IN PANDAS##########################

# Basic iteration (for i in object) produces:
# Series: values
# DataFrame: column labels

#To iterate over the rows of a DataFrame use:

# 1. iterrows(): Iterate over the rows of a DataFrame as (index, Series) pairs.
#  This converts the rows to Series objects, which can change the dtypes and
#  has some performance implications.
# 2. itertuples(): Iterate over the rows of a DataFrame as namedtuples of the
# values. This is a lot faster than iterrows(), and is in most cases preferable
# to use to iterate over the values of a DataFrame.


##Consistent with the dict-like interface, iteritems() iterates through key-value pairs:
# Series: (index, scalar value) pairs
# DataFrame: (column, Series) pairs

#########################APPENDING TO DataFrame##################################

#define a new row
#The .loc/[] operations can perform enlargement when setting a non-existent key for that axis

se = pd.Series([1,2,3])

se[5] = 5.

dfi = pd.DataFrame(np.arange(6).reshape(3,2),
 columns=['A','B'])
dfi.loc[:,'C'] = dfi.loc[:,'A']
#This is like an append operation on the DataFrame.

dfi.loc[3] = 5



#############OPERATION ON DATA STRUCTURES################
#we will discuss on dataframe only which shoulg apply on series too

# Elementwise NumPy ufuncs (log, exp, sqrt, ...) and various other NumPy
# functions can be used with no issues on DataFrame, assuming the data within
# are numeric

index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
#since the base of pandas is like numpy, we can do scalar operation
#that executes element-wise
df - df.iloc[0] #subtracts first row from all the cols
#for all operation

df.sub(df['A'], axis=0) #subtracts the col and broadcasts it
print df * 5 + 2
print 1 / df
print df ** 4

#boolean works too
df1 = pd.DataFrame({'a' : [1, 0, 1], 'b' : [0, 1, 1] }, dtype=bool)
df2 = pd.DataFrame({'a' : [0, 1, 1], 'b' : [1, 1, 0] }, dtype=bool)

print df1 & df2


#to get the transpose of data
print df.T
#this transposes the data with index and columns names
#The dot method on DataFrame implements matrix multiplication

print df.T.dot(df)


#fill values of nan during mathematical ops
df.add(df, fill_value=0)
# Series and DataFrame have the binary comparison methods
# eq, ne, lt, gt, le, and ge whose behavior is analogous
# to the binary arithmetic operations
#eq - equal
#ne - not equal
#lt - less than
#gt - greater than
#le - less (than or) equal (to)
#ge - greater (than or) equal (to)
print df.eq(df) #element wise comparison

#BOOLEAN REDUCTIONS
#we can reduce queries column-wise
(df > 0).all()
#check for each column and all values
#similary any and empty
# You can conveniently do element-wise comparisons
# when comparing a pandas data structure with a scalar value

print pd.Series(['foo', 'bar', 'baz']) == 'foo'

#COMBINATION OF TWO SIMILAR DATASET where values in calling df are
#preferred over the other but values from other df are used iff
#df1 has NaN in the same location in which df2 has a value

df1 = pd.DataFrame({'A' : [1., np.nan, 3., 5., np.nan],
 'B' : [np.nan, 2., 3., np.nan, 6.]})
df2 = pd.DataFrame({'A' : [5., 2., 4., np.nan, 3., 7.],
'B' : [np.nan, np.nan, 3., 4., 6., 8.]})

print df1.combine_first(df2)


##############################STATISTICS#####################################
#DESCRIPTIVE STATISTICS
# Most of these are aggregations (hence producing a lower-dimensional
# result) like sum(), mean(), and quantile(), but some of them, like
# cumsum() and cumprod(), produce an object of the same size. Generally
# speaking, these methods take an axis argument, just like
# ndarray.{sum, std, ...}, but the axis can be specified by name or integer

print df1.mean(0) #column-wise mean
print df1.mean(1) #row-wise mean

#All such methods have a skipna option signaling whether to exclude missing
#data (True by default)
#all functions on page 530 of docs

# The idxmin() and idxmax() functions on Series and DataFrame compute the index
# labels with the minimum and maximum corresponding values
s1 = pd.Series(np.random.randn(5))
print s1.idxmin(), s1.idxmax()
df1 = pd.DataFrame(np.random.randn(5,3), columns=['A','B','C'])
df1.idxmax(axis=1)

#Arbitrary functions can be applied along the axes of a DataFrame or Panel using
# the apply() method, which, like the descriptive statistics methods, take an optional
# axis argument
print df1.apply(np.mean,axis=1)

##################################AGGEREGATION API###################################################
# The aggregation API allows one to express possibly multiple aggregation operations in a
# single concise way.
tsdf = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'],
 index=pd.date_range('1/1/2000', periods=10))
# Using a single function is equivalent to apply(); You can also pass named methods as strings.
# These will return a Series of the aggregated output

print tsdf.agg(np.sum)
print tsdf.agg('sum')

#You can pass multiple aggregation arguments as a list
tsdf.agg(['sum', 'mean'])

#USING APPLY AND APPLYMAP for element-wise application of functions
# Since not all functions can be vectorized (accept NumPy arrays and return another array or value),
# the methods applymap() on DataFrame and analogously map() on Series accept any Python function taking
# a single value and returning a single value

#Percent changes
#a method pct_change to compute the percent change over a given number of periods (using fill_method to
#fill NA/null values before computing the percent change)

ser = pd.Series(np.random.randn(8))
print ser.pct_change()
#Covariance
#The Series object has a method cov to compute covariance between series (excluding NA/null values).

s1 = pd.Series(np.random.randn(1000))
s2 = pd.Series(np.random.randn(1000))

print s1.cov(s2)

#Correlation
#for Correlation between two DataFrames
#frame['a'].corr(frame['b'])
#for Correlation of all elements pairwise
#frame.corr()

#WINDOW function

#For working with data, a number of windows functions are provided for computing common window or rolling statistics
s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000',
periods=1000))
r = s.rolling(window=60)
print dir(r)
#view all the function it provides just for the moving window of 60 data point

#all the methods are given below these can be applied with rolling, aggregation
# count()- Number of non-null observations
# sum()- Sum of values
# mean()- Mean of values
# median() -Arithmetic median of values
# min()- Minimum
# max()- Maximum
# std()- Bessel-corrected sample standard deviation
# var()- Unbiased variance
# skew()- Sample skewness (3rd moment)
# kurt()- Sample kurtosis (4th moment)
# quantile()- Sample quantile (value at %)
# apply()- Generic apply
# cov()- Unbiased covariance (binary)
# corr()- Correlation (binary)
# more about window function in docs(pg 703)






#################################SORTING#################################
#1 by index - sort_index()
unsorted_df = df.reindex(index=['a', 'd', 'c', 'b'],
columns=['three', 'two', 'one'])

unsorted_df.sort_index()
#2 by values- sort_values()
df1 = pd.DataFrame({'one':[2,1,1,1],'two':[1,3,2,4],'three':[5,4,3,2]})
 df1.sort_values(by='two')



##############WORKING WITH TEXT IN PANDAS################################
#on page no 585 of pandas docs
# Series and Index are equipped with a set of string processing methods that make it easy to operate on
# each element of the array. Perhaps most importantly, these methods exclude missing/NA values automatically.
# These are accessed via the str attribute.

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

print s.str.lower()
print s.str.upper()
print  s.str.len()

idx = pd.Index([' jack', 'jill ', ' jesse ', 'frank'])
idx.str.strip()

#The string methods on Index are especially useful for cleaning up or transforming DataFrame columns.

df = pd.DataFrame(randn(3, 2), columns=[' Column A ', ' Column B '],
 index=range(3))
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
s2 = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
s2.str.split('_', expand=True)
#expand makes column

#INDEX WITH .str
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan,
'CABA', 'dog', 'cat'])
s.str[0]


##########################METHOD SUMMARY OF STRINGS#################################################
#on page number 596


#========================= Hierarchical/ MULTIINDEXING (confusion)=========================================
# think of MultiIndex as an array of tuples where each tuple is unique. A MultiIndex can be created from a list
# of arrays (using MultiIndex.from_arrays), an array of tuples (using MultiIndex.from_tuples), or a crossed set of
# iterables (using MultiIndex.from_product)

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
tuples = list(zip(*arrays))
print tuples
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print index

s = pd.Series(np.random.randn(8), index=index)
print s

#When you want every pairing of the elements in two iterables, it can be easier to use the MultiIndex. from_product function

iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
print pd.MultiIndex.from_product(iterables, names=['first', 'second'])

arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]

s = pd.Series(np.random.randn(8), index=arrays)


df = pd.DataFrame(np.random.randn(8, 4), index=arrays)
print df.head()

#Indexing is same as nested dicts

# a lot of interesting features are available covered in docs (pg. 667)


#################################### Handling Missing(NA) Data ##########################
#Missing values are generally NaN and na values
#NaN is default missing value arker for the reason of fast computation specified
#In many case python None will arise and we consider that as na
#nan!=nan nan!=None
# To make detecting missing values easier (and across different array dtypes), pandas provides
# the isna() and notna() functions, which are also methods on Series and DataFrame objects

s = pd.Series([1, 2, 3])
s.loc[0] = None
print s
print s.isna()
print s.notna()

##Inserting missing data
#The fillna function can “fill in” NA values with non-NA data in a couple of ways
s.fillna(0)  #filling with scalars

s.fillna('missing') #filling with str

#we can limit amount of filling
s.fillna(0,limit=1)
#this will fill only one na value if MULTIPLE CONSECUTIVE VALUES are present

#different filling option
# ________________________________________
#    Method        |         Action       |
# ========================================
#   pad / ffill    |  Fill values forward |
# bfill / backfill |  Fill values backward|
# =========================================

#filling with Mean (statistically the best option)
s.fillna(s.mean())
print s

#We can drop the missing data too, by dropping the complete index or column
#To do this, use the dropna method
s.dropna(axis=0) # ALERT here default axis, axis =0 is row-wise rather than column-wise
print s
#interpolation is another mathematical tool to fill missing values
#interpolation is basically filling in the data to make a smooth curve with all the data points
#more on interpolation in docs (page 737)

##############################GROUP BY: SPLIT-APPLY-COMBINE######################################

# "Group by” refers to a process involving one or more of the following steps
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure

#To create a GroupBy object
# grouped = obj.groupby(key)
# grouped = obj.groupby(key, axis=1)
# grouped = obj.groupby([key1, key2])

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
'foo', 'bar', 'foo', 'foo'],
'B' : ['one', 'one', 'two', 'three',
'two', 'two', 'one', 'three'],
'C' : np.random.randn(8),
'D' : np.random.randn(8)})

print df

grouped = df.groupby('A')
print grouped.first()

grouped = df.groupby(['A', 'B'])
print grouped.last()

#ALERT grouped.head() print df
#grouped.first() prints the first group members

#agg functions can be applied to groups

#group by sorting
#By default the group keys are sorted during the groupby operation. You may
#however pass sort=False for potential speedups

#The groups attribute is a dict whose keys are the computed unique groups
#and corresponding values being the axis labels belonging to each group

print df.groupby('A').groups

#Grouping data with Hierarchical index

arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
 ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

s = pd.Series(np.random.randn(8), index=index)
print s
grouped = s.groupby(level=0)

##ITERATING THROUGH GROUPS
for name, group in grouped:
    print(name)
    print(group)

#since grouping gives a collection of df objects all other function are applicable to
#individual groups
#read more on about it in docs page no. 773



###### MERGE JOIN AND CONCATENATE: INCREASING/APPENDING THE DATAFRAME##########
# The concat function (in the main pandas namespace) does all of the heavy lifting of
# performing concatenation operations along an axis while performing optional set logic
# (union or intersection) of the indexes (if any) on the other axes (Page 789)

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
'B': ['B0', 'B1', 'B2', 'B3'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']},
index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
'B': ['B4', 'B5', 'B6', 'B7'],
'C': ['C4', 'C5', 'C6', 'C7'],
'D': ['D4', 'D5', 'D6', 'D7']},
index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
'B': ['B8', 'B9', 'B10', 'B11'],
'C': ['C8', 'C9', 'C10', 'C11'], 'D': ['D8', 'D9', 'D10', 'D11']},
index=[8, 9, 10, 11])

frames = [df1, df2, df3]

result = pd.concat(frames)

#defualt pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
# keys=None, levels=None, names=None, verify_integrity=False,
# copy=True)

#we can pass the key param and it'll out index(Hierarchical) the concatenated dfs
result = pd.concat(frames, keys=['x', 'y', 'z'])
print result

#we can use axis argument too

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
'D': ['D2', 'D3', 'D6', 'D7'],
'F': ['F2', 'F3', 'F6', 'F7']},
 index=[2, 3, 6, 7])

result = pd.concat([df1, df4], axis=1)

#we can use join inner, more on join later
result = pd.concat([df1, df4], axis=1)
print result

#a useful shortcut is append works like concat on axis=0
result = df1.append(df2)
print result
#REMEMBER the indexes must be disjoint but the columns do not need to be
#append can take multiple argument
result = df1.append([df2, df3])
#For DataFrames which don’t have a meaningful index, we may wish to append
#them and ignore the fact that they may have overlapping indexes
#To do this, use the ignore_index argument
result = pd.concat([df1, df4], ignore_index=True)
result = df1.append(df4, ignore_index=True)

#concat series and DataFrame
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
result = pd.concat([df1, s1], axis=1)

#can append a single row to a DataFrame by
# passing a Series or dict to append, which returns a new DataFrame
s2 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])
result = df1.append(s2, ignore_index=True)
print result


####JOINS
# pandas has full-featured, high performance in-memory join operations idiomatically
# very similar to relational databases like SQL. These methods perform significantly better

#pandas provides a single function, merge, as the entry point for all standard database join
# operations between DataFrame objects
#default function pd.merge(left_df, right_df, how='inner', on=None, left_on=None, right_on=None,
# left_index=False, right_index=False, sort=True,
# suffixes=('_x', '_y'), copy=True, indicator=False,
# validate=None)
# for refreshing your memory on joins view docs page 803

#merge pandas vs SQL

#  Merge method |       SQL           |           Join Name Description          |
# ==============================================================================
#    left       | LEFT OUTER JOIN     |        Use keys from left frame only     |
#    right      | RIGHT OUTER JOIN    |        Use keys from right frame only    |
#    outer      | FULL OUTER JOIN     |      Use union of keys from both frames  |
#    inner      |   INNER JOIN        | Use intersection of keys from both frames|

#more on validating merge and indicating merge on page 806 of docs


#JOIN method
#DataFrame.join is a convenient method for combining the columns of two potentially
# differently-indexed DataFrames into a single result DataFrame

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
 'B': ['B0', 'B1', 'B2']},
index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
'D': ['D0', 'D2', 'D3']},
index=['K0', 'K2', 'K3'])
result = left.join(right) #it is default like pd.merge(left,right, how='left')
#join method takes in how arg too
result = left.join(right, how='outer')

#know about joining single index to multiindex in docs page no. 813

#A list or tuple of DataFrames can also be passed to DataFrame.join to join them together on their indexes
right2 = pd.DataFrame({'v': [7, 8, 9]}, index=['K1', 'K1', 'K2'])
result = left.join([right, right2])
print result


# =============================================================================
