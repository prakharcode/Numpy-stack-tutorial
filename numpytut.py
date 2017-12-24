import numpy as np

help(np.ndarray.ndim) #gives number of axes or total
help(np.ndarray.shape) #gives dimension of array
help(np.ndarray.size) #total number of elements
help(np.ndarray.dtype)
help(np.ndarray.itemsize)

#DATA TYPES IN NUMPY
np.int16
np.int32
np.float
np.bool
np.complex
np.str

a = np.arange(15).reshape(3,5)
print a
print a.shape
print a.ndim
print a.itemsize
print a.size
print type(a)

#Create an array

a = list((1,2,3,4))
a = np.array(a)
print a
#Indexing
print a[1]
print a[:]
# for 2d -arrays we can use a[i][j] or a[i,j]
print type(a)
c = np.array( [ [1,2], [3,4] ], dtype=complex ) #array with complex number
f = np.array([[1,2],[3,4]], dtype=float)# array with float dtype

#to create range of number np has arange method
#similar to range
r = np.arange(10,30)
#np.arange(start, end, jumps)
#to get equally spaced floating point number use linspace
n = np.linspace(0,2,10)
#np.linspace(start, stop, divisions)

#to generate zeros and ones use following commands

zero = np.zeros((10,10))
ones = np.ones((10,10))

#np.zeros(rows,cols)
#np.ones(rows,cols)

#to initialize empty array of a certain size
np.empty((10,10))
#np.empty((shape))


#to create array of zero similar to another array's size
np.zeros_like(a)
#np.zeros_like(array)
#similar function to generate ones
np.ones_like(a)

#to generate identity matrix
np.eye(3)
#np.eye(size)
#returns square matrix of shape: size x size

#to generate random array || DATA GENERATION
np.random.rand(10,10) #random number uniform distribution 0,1
np.random.randint(1,100,10) #random intergers between range of first and second arg
np.random.random((10,10)) # random similar to rand
np.random.ranf((10,10)) #random floats
np.random.randn(10,10) # random number from standard normal
np.random.normal(3,1,10) #random number from gaussaian distrib
                        #np.random.normal(mu,sigma,sample)

#to plot a histogram we can get data from np.histogram
(n, bins) = np.histogram(v, bins=50, normed=True) # NumPy version (no plot)
#normed means normal distribution

#  BASIC operations in numpy

'''
Arithemetic operations in numpy are applied in
elememt-wise fashion
A new array is created for the result.
'''
a = np.arange(10,20,5)
b = np.arange(20,30,5)


#Matrix subtraction
c = a-b
#Matrix summation
s = a+b

#element-wise power operations

a**2

#the * operation is used for ELEMENT WISE multiplication
p = a*b

#for matric multiplication use function dot


m = np.dot(a,b) #this is the inner product or dot product
m = a.dot(b)    #gives the smaller dimension out, here shape: 1x1
#can be used in any of the way mentioned above



m = np.outer(a,b) #this is the outer product
                  #gives the smaller dimension out, here shape: 5x5


#Since operations are carried out element wise ones
#can use the operators like +=, -=, /= and *=
a+=2
b+=a #b=b+a

#operations like '+=' are carried with data type hierarchy
#this means an int can be added to float but not
#the other way cause they are further assigned too

#but int and float can be added freely if they are to
#be inserted in new array

a = np.ones((3,3), dtype=np.int32)
b = np.linspace(0,pi,9).reshape(3,3)
c = a + b

#UNIARY OPERATION
#to get the sum of all the element

a = np.one((10,10))
sums =a.sum()

#to get sum column-wise
a.sum(axis=0)


#to get the minimum of all the elements
a.min()
#to get maximum of all the elements
a.max()
#to get the minimum column-wise
a.min(axis=0)
#to get the maximum column-wise
a.min(axis=0)

#to get the minimum row-wise
a.min(axis=1)
#to get the maximum row-wise
a.min(axis=1)

#to get cumulative sum along row
#cumulative sum is a[i]+a[i-1]

a.cumsum(axis=1)


#to get cumulative sum along column
#cumulative sum is a[i]+a[i-1]

a.cumsum(axis=0)


#UNIVERSAL FUNCTION

#since numpy is a elements wise mathematical operation
#library. It gives universal mathematical functions
#applicable to both single numbers and Matrix

np.sqrt(25) #to get sqrt
np.exp(2) #to get exponential of num or matrices
np.add(2,3) #to add two numbers or matrices
np.rad2deg(1.5)
np.radians(90)
#Apply a function over axis
#np.apply(func, array, axes)
np.apply_over_axes(np.sum, a,0)
#to apply to each element use apply

#to flatten a multi-dimensional array (make 1-D)
a.ravel()

# to take transpose of a Matrix
a.T

#resize v/s reshape
a.shape = 2,3 #changes a
a.resize((2,3)) #changes a
a.reshape(2,3) #returns an array of size 2 x 3

#a.reshape(2,-1) if given -1 the shape is calculated
#automatically


#STACKING DIFFERENT ARRAY
a =np.floor(np.random.random((10,10)))
b = np.floor(10*np.random.random((2,10)))
c = np.floor(10*np.random.random((10,2)))

#for stacking vertically
np.vstack((a,b))
#returns matrix of shape 12 x 10
#for stacking horizontally

np.hstack((a,b))
#returns matrix of shape 10 x 12

 #splitting arrays into multiple arrays

a = np.floor(10*np.random.random((10,12)))
#splitting horizontally by dividing index in 3
np.hsplit(a,3)
#return array of shape(4,10,3)
#splitting vertically

#splitting array based on their location
np.hsplit(a,(3,5)) #splits array from 0-3,3-5,5-10 in axes=0

np.vsplit(a,5)
np.vsplit(a,(3,5))
#return array of shape(2,5,12)


#COPIES
'''
In numpy general assigns does not generate copies
instead it generates a reference
'''
# NO COPY
a = np.arange(12)
b = a #b is referencing to a only not a copy
b.shape=3,4
b.shape == a.shape #True
b is a # =True


#SHALLOW COPY
b = a.view()
b is a # false
b.base is a #True

b.shape =3,4
b.shape == a.shape #false
b[:,1:2] = 10 #changes a

#COMPLETE NEW COPY / DEEP COPY

b = a.copy()
b is a # false
b.base is a #false
b[:,1:2] = 10 #doesn't change a

# Indexing of array with array

a = np.arange(12)**2
i = np.array([1,2,2,5,6,2])
print a[i] #returns an array consisting
#elements from a and index as mentioned in i
#The shape of the returned array will be same as of i

a[i].shape == i.shape #True

#a use case is given below

image = np.array( [ [ 0, 1, 2, 0 ],
[ 0, 3, 4, 0 ] ] )# each value corresponds to a color in the palette
palette[image] # the (2,4,3) color image
# array([[[ 0, 0, 0],
# [255, 0, 0],
# [ 0, 255, 0],
# [ 0, 0, 0]],
# [[ 0, 0, 0],
# [ 0, 0, 255],
# [255, 255, 255],
# [ 0, 0, 0]]])

# We can also use boolean Indexing
a = np.arange(12).reshape(3,4)
print a[a>4]

#LINEAR ALGEBRA
#numpy has linalg module to do linear algebric operation

a.transpose() # same as a.T
np.linalg.inv(a) #should be a square


#Readind Array from string
import StringIO from StringIO
data = "1, 2, 3\n 4, 5, 6"
np.genfromtxt(StringIO(data), delimiter=",")
data = " 1 2 3\n 4 5 67\n890123 4"
np.genfromtxt(StringIO(data), delimiter=3)

#SAVING NUMPY ARRAY
#np.save(file, arr, allow_pickle=True, fix_imports=True) #ONLY SAVES ONE ARRAY AT A TIME
#np.savez(file, *args, **kwds)  SAVES IN .npy FORMAT, MANY ARRAY CAN BE SAVED 
#np.savez_compressed(file, *args, **kwds) SAVES IN .npz FORMAT
#np.savetxt(fname, arr, fmt='%.18e', delimiter=' ', newline='\n',
# header='', footer='', comments='# ')



#LOADING NUMPY ARRAY
#np.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')


#Example
from tempfile import TemporaryFile
outfile = TemporaryFile()
x = np.arange(10)
y = np.sin(x)
np.savez(outfile, x, y)
outfile.seek(0) # Only needed here to simulate closing & reopening file
npzfile = np.load(outfile)
print npzfile.files
npzfile[npzfile.files[0]]
