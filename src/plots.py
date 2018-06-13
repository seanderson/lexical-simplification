'''
Various plotting functions.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def hist(fig,ax,dat,nbins,name,nplots,plotnum):
    '''Call inithist first.  Pass in fig and ax returned from inithist.
    plotnum is zero-based.'''
    #plt.subplot(nplots,1,plotnum) # select subplot
    # the histogram of the data
    n, bins, patches = ax[plotnum].hist(dat, nbins, normed=False, facecolor='red', alpha=0.75)

    # hist uses np.histogram under the hood to create 'n' and 'bins'.
    # np.histogram returns the bin edges, so there will be nbins
    # probability density values in n, nbinbs+1 bin edges in bins and
    # nbins patches.  To get everything lined up, we'll compute the bin
    # centers bincenters = 0.5*(bins[1:]+bins[:-1])

    ax[plotnum].set_ylabel(name)
    #ax[plotnum].set_ylabel('Count')

    lastVal = max(dat)
    firstVal = min(dat)
    axlen = lastVal - firstVal
    margin = axlen * 0.05
    ax[plotnum].set_xlim(firstVal - margin,lastVal+margin)
    #ax.set_ylim(0, 0.03)
    ax[plotnum].grid(False)
    plt.tight_layout(w_pad=0,h_pad=0)
    #plt.title(name)
    #plt.text(0,1,name)

    
def inithist(nplots):
    '''Returns fig and ax, the latter an array.'''
    fig,ax = plt.subplots(nrows=nplots,ncols=1)
    fig.set_size_inches(12,17,forward=True)
    return fig,ax

def main(): #debuggin
    nbins = 20
    nplots = 8
    fig,ax = inithist(nplots)
    for i in range(nplots):
        dat = np.random.uniform(-1,3,(100))
        hist( fig,ax,dat,nbins,"my plot",nplots,i)
    plt.show()

def show():
    plt.show()

if __name__ == "__main__":
    main()
