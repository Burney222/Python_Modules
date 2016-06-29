#Little functions for the daily life
import pandas as pd
import root_numpy as rn
import inspect
import numpy as np
import matplotlib.pyplot as plt

#Returns a pandas dataframe from a TTree (as alternative to ROOT pandas,
#can deal with functions as branches names)
def root2dataframe(*args, **kwargs):
    """
    Read root-file as pandas DataFrame.
    Usage: root2dataframe("filename", "treename"(optional), branches=...,
                          selection=...)
    """
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
def calc_overlap_integral(entries1, entries2, binning, weights1=None, weights2=None):
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

    hist1, _ = np.histogram(entries1, binning, density=True, weights=weights1)
    hist2, _ = np.histogram(entries2, binning, density=True, weights=weights2)

    #Take into account the entries that lay outside the range
    hist1_norm = np.sum(weights1) if weights1 is not None else len(entries1)
    hist2_norm = np.sum(weights2) if weights2 is not None else len(entries2)
    hist1 *= np.sum(np.histogram(entries1, binning, weights=weights1)[0])/hist1_norm
    hist2 *= np.sum(np.histogram(entries2, binning, weights=weights2)[0])/hist2_norm

    binwidths = binning[1:] - binning[:-1]

    #Calc and return the actual integral
    min_hist = np.minimum(hist1, hist2)
    return np.sum( min_hist * binwidths )



#Function to calculate efficiencies
def eff_w_error_dataset(dataset, cut, weights=None):
    """
    param dataset: dataframe to the tree where to apply the cut(s)
    param cut: cut to apply as string
    param weights (optional): weights column name in the dataset as string
    """
    before = np.sum(dataset[weights]) if weights is not None else float(len(dataset))
    after = ( np.sum( (dataset.query(cut, engine='python'))[weights] )
                                     if weights is not None
                                     else float(len(dataset.query(cut, engine='python'))) )

    eff = after/before
    eff_error = np.sqrt(eff*(1-eff)/before)

    return eff, eff_error


def eff_w_error(n_before, n_after):
    """
    n_before = entries before
    n_after = entries after
    """
    eff = n_after/n_before
    eff_error = np.sqrt(eff*(1-eff)/n_before)

    return (eff, eff_error)


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


#Get min/max values of given arrays (as if they were merged)
def min_max( arrays ):
    maxv = max( [max(array) for array in arrays if array is not None])
    minv = min( [min(array) for array in arrays if array is not None])
    return minv, maxv


#Plot classifier output
def Plot_Classifier_Output( sig_test, bkg_test, sig_train = None, bkg_train = None,
                            bins = None, normalised = True ):
    """
    sig/bkg_test: array of test-predictions
    sig/bkg_train: array of train-predictions (optional)
    bins: binning edges (optional)
    normalised: normalised signal/background (default: True)
    """
    #Binning
    if bins is not None:
        binning = bins
    else:
        minv, maxv = min_max( [sig_test, bkg_test, sig_train, bkg_train] )
        binning = np.linspace(minv, maxv, 51)

    #Plot test sample
    hist_errorbars(bkg_test, xerrs=True, bins = binning, normed=normalised, color='b',
                   label="Test Background")
    hist_errorbars(sig_test, xerrs=True, bins = binning, normed=normalised, color='r',
                   label="Test Signal")

    #Plot training sample if applicable
    if sig_train is not None or bkg_train is not None:
        if sig_train is None or bkg_train is None:
            raise ValueError("Both sig and bkg test-samples must be None or not None!")
        #Scale train to test-sample
        len_test = len(sig_test) + len(bkg_test)
        len_train = len(sig_train) + len(bkg_train)
        train_weight = len_test/len_train
        train_weight_sig = np.ones(len(sig_train))*train_weight
        train_weight_bkg = np.ones(len(bkg_train))*train_weight

        plt.hist(bkg_train, bins = binning, normed=normalised, color='b', weights=train_weight_bkg,
                alpha = 0.3, linewidth=0, hatch='\\\\\\\\\\' , label="Training Background")
        plt.hist(sig_train, bins = binning, normed=normalised, color='r', weights=train_weight_sig,
                alpha = 0.3, linewidth=0, hatch='/////', label="Training Signal")


    plt.xlabel("Classifier Output", fontsize=23)
    plt.ylabel("Normalised Events" if normalised else "Events", fontsize=23)

    #Plot settings
    plt.xlim(binning[0], binning[-1])
    plt.legend(loc='upper left', fontsize=19)



#Function to add a variable to an existing tree
def Add_to_TTree(input_filename, output_filename, var_array, var_name, var_type="D", treename=None):
    """
    input/output_filename: string - filename of root-files
    var_array: tuple with entries to store
    var_name: string - variable name where to save the variable
    var_type: string - variable type ('D' for double, 'I' for integer, ...)
    """
    #Open tree and clone it
    inputfile = ROOT.TFile(input_filename, "READ")
    if not inputfile.IsOpen():
        raise SystemExit("Could not open inputfile!")

    if treename == None:
        inputtreename = get_any_tree(input_filename)
    else:
        inputtreename = treename

    inputtree = inputfile.Get(inputtreename)

    entries = inputtree.GetEntries()
    #Consistency check
    if entries != len(var_array):
        raise SystemExit("Number of entries in {} does not match length of values to write to the "
                         "new file {}".format(input_filename, output_filename))

    #Clone tree
    outputfile =  ROOT.TFile(output_filename,"RECREATE")
    outputtree = inputtree.CloneTree(-1, "fast")

    #Create new branch
    index_branch = array.array(var_type.lower(), [0])
    new_branch = outputtree.Branch( var_name, index_branch, var_name+"/"+var_type )

    #Fill new branch
    print("Processing {0} entries in {2} /{1}".format(entries, inputtreename, input_filename))
    for i,value in  zip(trange(entries), var_array):
        index_branch[0] = value

        #Fill tree
        new_branch.Fill()

    print("Finished processing {0} entries in {2} /{1}\n   => Writing to {3}"
          "".format(entries, inputtreename, input_filename, output_filename))
    outputtree.Write()
    outputfile.Close()
