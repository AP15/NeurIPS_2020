import numpy as np, pandas as pd
from oracle import SCQOracle

class LabelSampler:
    """
    Takes labeled samples in a pipeline fashion.

    This class supports sampling u.a.r. from a specified dataset until one of the labels
    has enough samples (specified by a threshold) at which point samples are returned.
    The dataset is passed at every sampling call, and the sampler drops any old samples that
    are not in the specified dataset.

    Examples
    --------
    # 10 points, 2 classes, 3 samples each round
    X = pd.DataFrame({'label': np.random.randint(2,size=10)})
    X
       label
    0      0
    1      0
    2      1
    3      1
    4      0
    5      1
    6      1
    7      0
    8      1
    9      1
    oc = SCQOracle(pd.DataFrame(X['label']))
    sampler = LabelSampler(oc, 3)
    sampler.sample(X)
    (0, [0, 1, 4])
    sampler.sample(X.drop([0,1,4]))
    (1, [8, 5, 6])
    sampler.sample(X.drop([0,1,4,8,5,6]))
    (1, [9, 2, 3])
    sampler.sample(X.drop([0,1,4,8,5,6,9,2,3]))
    (0, [7])
    """

    def __init__(self, oc: SCQOracle, t: int = 1):
        """

        Parameters
        ----------
        oc: SCQOracle
            an oracle that supports label() queries
        t: int
            label sample size threshold

        Builds a sampler that will return samples as soon as some label get samples ≥ t times.
        """
        self.oc = oc
        self.samples = set()  # plain samples
        self.sample_dict = {}  # each label to its samples
        self.t = t  # default threshold
        self.ms_l = -1  # the class with most samples
        self.ms = -1  # the samples in class self.ms_l

    def update_largest_sample(self):
        """For internal use.
        """
        mss = [len(s) for s in self.sample_dict.values()]
        if mss:
            self.ms_l = np.argmax(mss)
            self.ms = mss[self.ms_l]
        else:
            self.ms = self.ms_l = -1

    def drop_samples(self, ds: pd.DataFrame):
        """Drop all samples that are not in the specified dataset. For internal use.

        Parameters
        ----------
        ds: DataFrame containing only the points to sample from (e.g. unlabeled ones)
        """
        for lab in self.sample_dict:
            for idx in self.sample_dict[lab].copy():
                if idx not in ds.index:
                    self.sample_dict[lab].remove(idx)
                    self.samples.remove(idx)
        self.update_largest_sample()

    def sample(self, ds: pd.DataFrame, t: int = None):
        """Sample t points with the same label from a dataset.

        This method takes samples from a dataset until some label appears
        in at least t samples. That label and its points are then returned.
        If dataset is not large enough then points will be returned earlier.

        Parameters
        ----------
        ds: DataFrame
            points to sample from
        t: threshold
            overrides the constructor (see its documentation)

        Returns
        -------
        (l, S): tuple
            l is the label, S is the list of samples (their index in the dataframe)
        """
        if self.ms > 0:
            self.drop_samples(ds)
        if t is None:
            t = self.t
        while self.ms < t and len(self.samples) < ds.shape[0]:
            x = ds.sample(1).index[0]  # take one sample
            if x in self.samples:
                continue  # was already taken
            lab = self.oc.label(x)
            if lab in self.sample_dict:  # add sample
                self.sample_dict[lab].add(x)
            else:
                self.sample_dict[lab] = {x}
            if len(self.sample_dict[lab]) > self.ms:  # update the most-sampled class
                self.ms = len(self.sample_dict[lab])
                self.ms_l = lab
            self.samples.add(x)
        # pop sample and return it
        l, s = self.ms_l, self.sample_dict[self.ms_l]
        del self.sample_dict[l]  # forget these samples
        self.samples.difference_update(s)  # forget these samples
        self.update_largest_sample()
        return l, list(s)  # return the most-sampled class
