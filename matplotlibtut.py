# Matplotlib is a library for making 2D plots of arrays in Python.
# Although it has its origins in emulating the MATLAB graphics
# commands, it is independent of MATLAB, and can be used in a
# Pythonic, object oriented way. Although Matplotlib is written
# primarily in pure Python, it makes heavy use of NumPy and
# other extension code to provide good performance even for
# large arrays

#The matplotlib.pyplot module contains functions that allow you to
#generate many kinds of plots quickly.

#various useful functions
#given in docs on page 339

import numpy as np
import matplotlib.pyplot as plt
#matplotlib is a very basic api to use

#the plots that you may need are
help(plt.plot)
help(plt.scatter)
help(plt.hist)

#plot is a method for creating lines it accepts 2 arrays first being
#x-axis and second being y, it also accepts a string like 'r--' where
#first character specifies the color, red here and the following characters
#line type, dashed here.
#to view all the properties go here: https://matplotlib.org/tutorials/introductory/pyplot.html#controlling-line-properties

#scatter(x_array, y_array, s=None, c=None, marker=None, cmap=None) this creates a scatter
#plot with x array and y array

#hist(x, bins=None, range=None) this plots a histogram of distribution
#takes in x array, bin size and bin range

#Working with multiple figures and axes
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

#one can alsp use np.linspace to get x range for function, useful for function graphing

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

#here we are plotting 2 plot in a single figures
#figures can be thought of as individual pop up windows
#contains plot and subplot specifies the location of the
#plot in the figure
#plt.subplot(211) specifies the subplot will have 2 rows
#1 col and this plot will be the 1 plot
#plt.subplot('n_rows'+'n_col'+'Index_figure')

#Working with text in the a plot

# The text() command can be used to add text in an arbitrary
# location, and the xlabel(), ylabel() and title() are used to
# add text in the indicated locations

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()
#notice the text option supports unicode encoding

#t = plt.xlabel('my data', fontsize=14, color='red')
#for more customizations and technique look into
#https://matplotlib.org/tutorials/text/text_props.html#sphx-glr-tutorials-text-text-props-py


#Images in matplotlib
import matplotlib.image as mpimg
# Loading image data is supported by the Pillow library. Natively, matplotlib only supports PNG images.
# The commands fall back on Pillow if the native read fails

# img = mpimg.imread('filename')
# print(img)
#the img is a numpy array having the pixel value for each pixel

#Plotting an image

#plt.imshow(img)
#implicit cmap application
# plt.imshow(lum_img, cmap="hot")
#explicit cmap application
# imgplot = plt.imshow(lum_img)
# imgplot.set_cmap('nipy_spectral')
#the cmap specifies which color map to use
#to view every color map available refer to documentation

#to get full details on how to manipuate images using Pillow(PIL)
#refer to PIL docs

#to Save a plotted image use Matplotlib.figure.Figure.savefig(fname, **kwargs)


#that's about all that you need to know about Matplotlib
#for experimenting with new figure and charts refer to the gallery of Matplotlib
#https://matplotlib.org/gallery/index.html#
