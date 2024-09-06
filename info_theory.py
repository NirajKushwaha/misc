from utils import *

def data_entropy(x, normalized=False):
    """
    Calculate shannon entropy for a series data.
    
    Parameters
    ----------
    x : list like
    
    Returns
    -------
    float
    """
    
    counts = Counter(x)

    entropy = 0
    for i, count in counts.items():
        entropy += (count/len(x)) * np.log2((count/len(x)))

    if(normalized):
        return -entropy/np.log2(len(counts))
    else:
        return -entropy
