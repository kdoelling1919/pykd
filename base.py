"""

Create useful tools for doing things in python

"""

from matplotlib import plot as plt
import numpy as np


def bar(x=None,y=None,**kwargs):
    # checking y as an input
    if y is None:
        raise TypeError('Y must have values for us to plot.')
    elif not isinstance(y,np.ndarray):
        raise TypeError('y must be of type np.ndarray. Instead it is {0}'.format(type(y)))
    elif len(y.shape) > 2:
        raise ValueError('y must have no more than 2 dimensions, instead it has {0}'.format(len(y.shape)))
    elif len(y.shape) < 2:
        y = y[:,None]

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

    h = list()
    for num,group in enumerate(y):
        ob=plt.bar(x+num*width,y,width=width,**kwargs)
        h.append(ob)

    tick_add = (grp_space-grp_margin)/2.0
    xticks = x+tick_add
    ax = plt.gca()

    ax.set_xticks(xticks)
    ax.set_xticklabels(x_)

    return ax,h
