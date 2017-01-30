"""

Create useful tools for doing things in python

"""

import matplotlib.pyplot as plt
import numpy as np


def bar(x=None,y=None,yerr=None,**kwargs):
    # checking y as an input
    if y is None:
        raise TypeError('Y must have values for us to plot.')
    elif not isinstance(y,np.ndarray):
        raise TypeError('y must be of type np.ndarray. Instead it is {0}'.format(type(y)))
    elif len(y.shape) > 2:
        raise ValueError('y must have no more than 2 dimensions, instead it has {0}'.format(len(y.shape)))
    elif len(y.shape) < 2:
        y = y[:,None]
    # checking yerr
    if yerr is None:
        yerr is np.zeros(y.shape)

    assert np.array_equal(yerr.shape,y.shape)

    # checking input for x
    if x is None:
        x = np.arange(y.shape[0])
        x_ = map(str,x)
    elif isinstance(x,(list,tuple)):
        x_=map(str,x)
        x = np.arange(len(x_))
    elif isinstance(x,np.ndarray):
        if len(x.shape) > 1:
            raise ValueError('x must be single dimensional')
        else:
            x_=map(str,x)
    else:
        raise TypeError('Cannot handle this input for x')

    assert x.shape[0] == y.shape[0]

    # get the bars parameters
    barsingroup = y.shape[1]
    grp_space = np.min(np.diff(x))
    grp_margin = .1*grp_space

    if not 'width' in kwargs:
        width = (grp_space -grp_margin)/barsingroup
    else:
        width = kwargs.pop('width')

    if 'color' in kwargs:
        color = kwargs.pop('color')
        assert len(color) == y.shape[0]


    h = list()
    for num,group in enumerate(y):
        ob=plt.bar(x+num*width,group,yerr=yerr[num],width=width,color=color[num],**kwargs)
        h.append(ob)

    tick_add = (grp_space-grp_margin)/2.0
    xticks = x+tick_add
    ax = plt.gca()

    ax.set_xticks(xticks)
    ax.set_xticklabels(x_)
    ax.set_xlim([x[0]-.5*width,x[-1]+(num+1.5)*width])

    return ax,h

def axes_fontsize(ax=None,fs=12):
    from matplotlib import axes
    # check axes input
    if ax is None:
        ax = plt.gca()
    elif not isinstance(ax,axes.Axes):
        raise TypeError('Input must be an Axes handle')

    # check fontsize
    if not isinstance(fs,int):
        raise TypeError('fs should be an int. Instead, it was {}'.format(type(fs)))
    elif fs <= 0:
        raise ValueError('fs should be positive. Instead, it was {}'.format(fs))

    ax=plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fs)
