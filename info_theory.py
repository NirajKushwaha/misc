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

    if(len(counts) == 1):  ## To avoid division by log(0) error in normalized entropy.
        return 0
    else:
        entropy = 0
        for i, count in counts.items():
            entropy += (count/len(x)) * np.log2((count/len(x)))

        if(normalized):
            return -entropy/np.log2(len(counts))
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
    
    Parameters
    ----------
    time_series_1 : list like
    time_series_2 : list like
    bootstrap : bool, False
    num_of_trials : int, 1000
    
    Instance Attributes
    -------------------
    mutual_info : float
    mutual_info_trials : list of float
    """

    def __init__(self, time_series_1, time_series_2, bootstrap=False, num_of_trials=1000):
        self.time_series_1 = time_series_1
        self.time_series_2 = time_series_2
        self.bootstrap = bootstrap
        self.num_of_trials = num_of_trials
        
        self.mutual_info = self._calculate_mutual_information(self.time_series_1, self.time_series_2)

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
        
        if np.array_equal(self.time_series_1, time_series_1) and np.array_equal(self.time_series_2, time_series_2):
            self.joint_prob = joint_prob

        prob_time_series_1 = Counter(time_series_1)
        prob_time_series_2 = Counter(time_series_2)

        mutual_info = 0.0

        for (i, j), prob_joint in joint_prob.items():
            prob_x = prob_time_series_1[i] / total_events
            prob_y = prob_time_series_2[j] / total_events
            mutual_info += prob_joint * np.log2(prob_joint / (prob_x*prob_y))

        return mutual_info

    def _bootstrap(self):
        """
        Bootstrap the mutual information calculation.

        Returns
        -------
        list of float
        """

        mutual_info_trials = []
        for _ in range(self.num_of_trials):
            time_series_1_boot = np.random.choice(self.time_series_1, len(self.time_series_1), replace=True)
            time_series_2_boot = np.random.choice(self.time_series_2, len(self.time_series_2), replace=True)

            mutual_info_boot = self._calculate_mutual_information(time_series_1_boot, time_series_2_boot)
            mutual_info_trials.append(mutual_info_boot)
        
        return mutual_info_trials