#Little functions for the daily life
import pandas as pd
import root_numpy as rn
import inspect
import numpy as np
import matplotlib.pyplot as plt

#Returns a pandas dataframe from a TTree (as alternative to ROOT pandas,
#can deal with functions as branches names)
def root2dataframe(*args, **kwargs):
    return pd.DataFrame(rn.root2array(*args, **kwargs))




#Function to return name of a tree if only one tree is in the given rootfile
def get_any_tree(tfilepath):
    trees = rn.list_trees(tfilepath)   #returns list of tree-names
    if len(trees) == 1:
        tree_name = trees[0]
    else:
        raise ValueError('More than one tree found in {}'.format(tfilepath))

    return tree_name




#Calculate the overlag integral of two distributions
def calc_overlap_integral(entries1, entries2, binning):
    """
    param entries1/2: are arrays of entries of the variables to plot
    param binning: is an array of bin-edges or an integer for the number of bins
    return: Overlap integral (value from 0 (well separated) to 1 (total overlap))
    """
    if len(binning) == 1:
        minv = min( min(entries1), min(entries2) )
        maxv = max( max(entries1), max(entries2) )
        binning = np.linspace(minv, maxv, int(binning)+1)
        print("Binning given as scalar: Using min and max of entries as ranges.")

    hist1, _ = np.histogram(entries1, binning, density=True)
    hist2, _ = np.histogram(entries2, binning, density=True)

    #Take into account the entries that lay outside the range
    hist1 *= np.sum(np.histogram(entries1, binning)[0])/len(entries1)
    hist2 *= np.sum(np.histogram(entries2, binning)[0])/len(entries2)

    binwidths = binning[1:] - binning[:-1]

    #Calc and return the actual integral
    min_hist = np.minimum(hist1, hist2)
    return np.sum( min_hist * binwidths )



#Function to calculate efficiencies
def eff_w_error(dataset, cut, weights=None):
    """
    param dataset: dataframe to the tree where to apply the cut(s)
    param cut: cut to apply as string
    param weights (optional): weights column name in the dataset as string
    """
    before = np.sum(dataset[weights]) if weights else float(len(dataset))
    after = np.sum( (dataset.query(cut))[weights] ) if weights else float(len(dataset.query(cut)))

    eff = after/before
    eff_error = np.sqrt(eff*(1-eff)/before)

    return eff, eff_error



#Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar
def hist_errorbars( data, xerrs=True, *args, **kwargs) :
    """Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar"""
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if 'normed' in kwargs.keys() :
        norm = kwargs.pop('normed')

    # pop off weights kwarg, since we want to handle it specially
    weights = None
    if 'weights' in kwargs.keys() :
        weights = kwargs.pop('weights')

    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getargspec(np.histogram).args :
            histkwargs[key] = value

    histvals, binedges = np.histogram( data, **histkwargs ) #without weights and normed
    yerrs = np.sqrt(histvals)

    if norm :
        nevents = float(sum(histvals))
        binwidth = (binedges[1]-binedges[0])
        histvals = histvals/nevents/binwidth
        yerrs = yerrs/nevents/binwidth

    if weights is not None:
        histvals_noweights = histvals
        histvals, binedges = np.histogram( data, weights=weights, **histkwargs)
        #Scale yerrs
        yerrs = yerrs.astype(float) * histvals.astype(float)/histvals_noweights.astype(float)

    bincenters = (binedges[1:]+binedges[:-1])/2

    if xerrs :
        xerrs = (binedges[1]-binedges[0])/2
    else :
        xerrs = None

    # retrieve the kwargs for errorbar
    ebkwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getargspec(plt.errorbar).args :
            ebkwargs[key] = value
        if key == 'color':
            ebkwargs['ecolor'] = value
            ebkwargs['mfc'] = value
        if key == 'label':
            ebkwargs['label'] = value

    out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt="o", mec='black', ms=8, **ebkwargs)


    if 'log' in kwargs.keys() :
        if kwargs['log'] :
            plt.yscale('log')

    if 'range' in kwargs.keys() :
        plt.xlim(*kwargs['range'])

    return out
