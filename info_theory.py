from .utils import *
try:
    from numba import njit
except:
    pass

def data_entropy(x, normalized=False, base="e"):
    """
    Calculate shannon entropy for a series data.

    Parameters
    ----------
    x : list like
    normalized : bool, False
    base : int/"e", "e"
        "e" for entropy in nats and 2 for entropy in bits.

    Returns
    -------
    float
    """

    counts = Counter(x)

    if(len(counts) == 1):  ## To avoid division by log(0) error in normalized entropy.
        return 0
    else:
        entropy = 0
        for i, count in counts.items():
            if(base == "e"):
                entropy += (count/len(x)) * np.log(count/len(x))
            else:
                entropy += (count/len(x)) * np.emath.logn(base, (count/len(x)))
    
        if(normalized):
            if(base == "e"):
                return -entropy/np.log(len(counts))
            else:
                return -entropy/np.emath.logn(base, len(counts))
        else:
            return -entropy

def normalized_mutual_information(time_series_1, time_series_2):
    """
    Calculates normalized mutual information (geometric) between two discrete time series.

    Parameter
    ---------
    time_series_1 : list like
    time_series_2 : list like
    
    Returns
    -------
    float
    """
    
    assert len(time_series_1) == len(time_series_2)
    
    joint_events = list(zip(time_series_1, time_series_2))

    joint_prob = Counter(joint_events)
    total_events = len(joint_events)

    prob_time_series_1 = Counter(time_series_1)
    prob_time_series_2 = Counter(time_series_2)

    mutual_info = 0.0

    for (i, j), count in joint_prob.items():
        prob_joint = count / total_events

        prob_x = prob_time_series_1[i] / total_events
        prob_y = prob_time_series_2[j] / total_events
        mutual_info += prob_joint * np.log2(prob_joint / (prob_x*prob_y))

    entropy_1 = data_entropy(time_series_1)
    entropy_2 = data_entropy(time_series_2)

    normalized_mutual_info = mutual_info / np.sqrt(entropy_1*entropy_2)

    return normalized_mutual_info

def joint_entropy(time_series_1, time_series_2):
    """
    Calculate joint shannon entropy of series data.
    
    Parameters
    ----------
    time_series_1 : list like
    time_series_2 : list like
    
    Return
    ------
    float
    """
    
    joint_events = list(zip(time_series_1, time_series_2))

    joint_prob = Counter(joint_events)
    total_events = len(joint_events)

    joint_entropy = 0.0

    for (i, j), count in joint_prob.items():
        prob_joint = count / total_events

        joint_entropy += prob_joint * np.log2(prob_joint)

    return -joint_entropy

def conditional_entropy(time_series_1, time_series_2, normalized=False):
    """
    Calculates conditional entropy for time_series_2 given time_series_1. 
    
    Parameter
    ---------
    time_series_1 : list like
    time_series_2 : list like
    normalized : bool, False
    
    Returns
    -------
    float
    """
    
    assert len(time_series_1) == len(time_series_2)
    
    joint_events = list(zip(time_series_1, time_series_2))

    joint_prob = Counter(joint_events)
    total_events = len(joint_events)

    prob_time_series_1 = Counter(time_series_1)
    prob_time_series_2 = Counter(time_series_2)

    cond_entro = 0.0

    for (i, j), count in joint_prob.items():
        prob_joint = count / total_events

        prob_x = prob_time_series_1[i] / total_events
        cond_entro += prob_joint * np.log2(prob_joint / prob_x)

    if(normalized):
        return - cond_entro / data_entropy(time_series_2)
    else:
        return - cond_entro

def KL_divergence_prob(P, Q):
    """
    Calculates KL divergence between discrete probability distributions P and Q (P||Q).

    Parameters
    ----------
    P : ndarray
    Q : ndarray

    Returns
    -------
    float
    """

    return np.sum(P * np.log(P / Q))

def JS_divergence_prob(P, Q):
    """
    Calculate Jensen-Shannon entropy between discrete probability distributions P and Q.

    Parameters
    ----------
    P : ndarray
    Q : ndarray

    Returns
    -------
    float
    """

    mixture = 0.5 * (P + Q)
    return 0.5 * (KL_divergence_prob(P, mixture) + KL_divergence_prob(Q, mixture))


class MutualInformation:
    """
    Class to calculate mutual information between two discrete time series.
    Currently supports MI calculation using frequency based method only.
    Estimation methods will be added in future.
    
    Instance Attributes
    -------------------
    mutual_info : float
    mutual_info_trials : list of float
    joint_prob : dict
    """

    def __init__(self, time_series_1, time_series_2, 
                 bootstrap=False, 
                 num_of_trials=1000, 
                 parallel_bootstrap=False,
                 multiprocess_chucksize=100):
        """
        Parameters
        ----------
        time_series_1 : list like
        time_series_2 : list like
        bootstrap : bool, False
        num_of_trials : int, 1000
        parallel_bootstrap : bool, False
        multiprocess_chucksize : int, 100
        """

        self.time_series_1 = time_series_1
        self.time_series_2 = time_series_2
        self.bootstrap = bootstrap
        self.num_of_trials = num_of_trials
        self.parallel_bootstrap = parallel_bootstrap
        self.multiprocess_chucksize = multiprocess_chucksize
        
        self.mutual_info, self.joint_prob = self._calculate_mutual_information(self.time_series_1, self.time_series_2)

        if(self.bootstrap):
            self.mutual_info_trials = self._bootstrap()

    def _calculate_mutual_information(self, time_series_1, time_series_2):
        """
        Calculate mutual information between two discrete time series.

        Parameters
        ----------
        time_series_1 : list like
        time_series_2 : list like

        Returns
        -------
        float
        """

        assert len(time_series_1) == len(time_series_2)

        joint_events = list(zip(time_series_1, time_series_2))
        total_events = len(joint_events)

        joint_prob = Counter(joint_events)
        joint_prob = {k:v/total_events for k, v in joint_prob.items()}

        prob_time_series_1 = Counter(time_series_1)
        prob_time_series_2 = Counter(time_series_2)

        mutual_info = 0.0

        for (i, j), prob_joint in joint_prob.items():
            prob_x = prob_time_series_1[i] / total_events
            prob_y = prob_time_series_2[j] / total_events
            mutual_info += prob_joint * np.log2(prob_joint / (prob_x*prob_y))

        if np.array_equal(self.time_series_1, time_series_1) and np.array_equal(self.time_series_2, time_series_2):
            return mutual_info, joint_prob
        else:
            return mutual_info

    def _bootstrap(self):
        """
        Bootstrap the mutual information calculation.

        Returns
        -------
        list of float
        """

        if(self.parallel_bootstrap):
            def looper(_):
                time_series_1_boot = np.random.choice(self.time_series_1, len(self.time_series_1), replace=True)
                time_series_2_boot = np.random.choice(self.time_series_2, len(self.time_series_2), replace=True)

                mutual_info_boot = self._calculate_mutual_information(time_series_1_boot, time_series_2_boot)
                return mutual_info_boot
            
            with Pool() as pool:
                mutual_info_trials = pool.map(looper, range(self.num_of_trials), 
                                              chunksize=self.multiprocess_chucksize)
            
            return mutual_info_trials
        else:
            mutual_info_trials = []
            for _ in range(self.num_of_trials):
                time_series_1_boot = np.random.choice(self.time_series_1, len(self.time_series_1), replace=True)
                time_series_2_boot = np.random.choice(self.time_series_2, len(self.time_series_2), replace=True)

                mutual_info_boot = self._calculate_mutual_information(time_series_1_boot, time_series_2_boot)
                mutual_info_trials.append(mutual_info_boot)
            
            return mutual_info_trials

# ======================================== #
# Class and functions for Transfer entropy #
# ======================================== # 

class TransferEntropy():
    def __init__(self, delay=1, rng=None):
        """
        Parameters
        ----------
        delay : int, 1
            Time delay between source x[t] and target future y[t + delay].
        rng : np.random.RandomState, None
        """

        if not isinstance(delay, (int, np.integer)):
            raise TypeError("delay must be an integer")
        if delay < 1:
            raise ValueError("delay must be a positive integer")

        self.important_indexes = [(0,0,0,0),(1,0,1,0),(2,1,2,1),(3,1,3,1),(4,2,0,0),(5,2,1,0),(6,3,2,1),(7,3,3,1)]
        self.delay = int(delay)
        self.rng = np.random if rng is None else rng

    def calc(self, x, y):
        """Calculate transfer entropy between two binary time series.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        float
            TE in bits.
        """
        
        assert set(np.unique(x)) <= frozenset((0,1)), "x must be binary"
        assert set(np.unique(y)) <= frozenset((0,1)), "y must be binary"
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if x.size <= self.delay:
            raise ValueError("time series length must be greater than delay")

        time_series = np.vstack((x, y)).T
        
        # calculate each term in the transfer entropy equation and sum them together
        joint_distribution_yT_yt_xt_list = joint_distribution_yT_yt_xt_new(time_series, self.delay)
        joint_distribution_yT_yt_list = joint_distribution_yT_yt_new(time_series, self.delay)
        joint_distribution_yt_xt_list = joint_distribution_yt_xt_new(time_series, self.delay)
        distribution_y_list = distribution_y_new(time_series, self.delay)

        te_terms = []

        for i, j, k, l in self.important_indexes:
            if(joint_distribution_yT_yt_xt_list[i] == 0 or
               joint_distribution_yT_yt_list[j] == 0 or
               joint_distribution_yt_xt_list[k] == 0 or
               distribution_y_list[l] == 0):
                pass
            else:
                te_terms.append(te_calculator((joint_distribution_yT_yt_xt_list[i],
                                               joint_distribution_yT_yt_list[j],
                                               joint_distribution_yt_xt_list[k],distribution_y_list[l])))
        return sum(te_terms)
    
    def shuffle_calc(self, x, y):
        """Shuffle both time series in the same way and return naive transfer entropy estimate.

        Returns
        -------
        float
            TE in bits.
        """

        randix = self.rng.permutation(np.arange(x.size, dtype=int))
        return self.calc(x[randix], y[randix])
# end TransferEntropy


@njit
def distribution_y_new(time_series_two_pols, delay=1):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_y = time_series_two_pols[:,1]

    time_series_y = time_series_y[:-delay]

    time_series_length = len(time_series_y)

    for i in range(len(time_series_y)):
        if(time_series_y[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_y[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array

@njit()
def distribution_x_new(time_series_two_pols, delay=1):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_y = time_series_two_pols[:,0]

    time_series_y = time_series_y[:-delay]

    time_series_length = len(time_series_y)

    for i in range(len(time_series_y)):
        if(time_series_y[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_y[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array

@njit()
def joint_distribution_yt_xt_new(time_series_two_pols, delay=1):
    counts_array = np.zeros(4)   #(yt,xt)  0->00 , 1->01 , 2->10 , 3-> 11

    time_series_yt_xt = time_series_two_pols[:-delay]

    time_series_length = len(time_series_yt_xt)

    for i in range(time_series_length):           
        if(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 1):
            counts_array[3] = counts_array[3] + 1

    distribution_array = np.zeros(4)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / time_series_length

    return distribution_array

@njit()
def joint_distribution_yT_yt_new(time_series_two_pols, delay=1):
    counts_array = np.zeros(4)   #(yT,yt)  0->00 , 1->01 , 2->10 , 3-> 11

    time_series_yT_yt = time_series_two_pols
    time_series_length = len(time_series_yT_yt)


    for i in range(time_series_length-delay):           
        if(time_series_yT_yt[i+delay][1] == 0 and time_series_yT_yt[i][1] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yT_yt[i+delay][1] == 0 and time_series_yT_yt[i][1] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yT_yt[i+delay][1] == 1 and time_series_yT_yt[i][1] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yT_yt[i+delay][1] == 1 and time_series_yT_yt[i][1] == 1):
            counts_array[3] = counts_array[3] + 1

    distribution_array = np.zeros(4)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / (time_series_length-delay)

    return distribution_array

@njit
def joint_distribution_yT_yt_xt_new(time_series_two_pols, delay=1):
    counts_array = np.zeros(8)   #(yT,yt,xt)  0->000,1->001,2->010,3->011,4->100,5->101,6->110,7->111

    time_series_yT_yt_xt = time_series_two_pols
    time_series_length = len(time_series_yT_yt_xt)

    for i in range(time_series_length-delay):           
        if(time_series_yT_yt_xt[i+delay][1] == 0 and
           time_series_yT_yt_xt[i][1] == 0 and
           time_series_yT_yt_xt[i][0] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 0 and
             time_series_yT_yt_xt[i][1] == 0 and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 0 and
             time_series_yT_yt_xt[i][1] == 1 and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 0  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[3] = counts_array[3] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 1  and
             time_series_yT_yt_xt[i][1] == 0  and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[4] = counts_array[4] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 1  and
             time_series_yT_yt_xt[i][1] == 0  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[5] = counts_array[5] + 1
        elif(time_series_yT_yt_xt[i+delay][1] == 1  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[6] = counts_array[6] + 1              
        elif(time_series_yT_yt_xt[i+delay][1] == 1  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[7] = counts_array[7] + 1            

    distribution_array = np.zeros(8)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / (time_series_length-delay)

    return distribution_array

@njit
def te_calculator(distribution_data):
    yT_yt_xt , yT_yt , yt_xt , y = distribution_data
    return (yT_yt_xt * np.log2((yT_yt_xt * y) / (yT_yt * yt_xt)))