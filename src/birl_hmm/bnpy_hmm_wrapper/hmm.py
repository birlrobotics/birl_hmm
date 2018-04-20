import bnpy
import numpy as np
import ipdb

class HongminHMM(object):
    def __init__(
        self, 
        alloModel,
        obsModel,
        varMethod,
        n_iteration,
        K,
        nTask = 1,                 # Number of runs to perform for single experiment
        nBatch = 10,               # Number of batches (aka minibatches) to split up dataset into.
        convergethr = 0.000000001, # for memoVB
        alpha = 0.5,
        gamma = 5.0,
        transAlpha = 5.0,
        startAlpha = 10.0,
        hmmKappa = 50.0,
        sF = 1.0,
        ECovMat = 'eye',
        initname = 'randexamples'):

        self.alloModel = alloModel
        self.obsModel = obsModel
        self.varMethod = varMethod
        self.n_iteration = n_iteration
        self.nTask = nTask
        self.nBatch = nBatch
        self.convergethr = convergethr
        self.alpha = alpha
        self.gamma = gamma
        self.transAlpha = transAlpha
        self.startAlpha = startAlpha
        self.hmmKappa = hmmKappa
        self.sF = sF
        self.ECovMat = ECovMat
        self.K = K
        self.initname = initname

    def fit(self, X, lengths):
        Xprev      = X[:-1,:]
        X          = X[1:,:]
        doc_range  = list([0])
        doc_range += (np.cumsum(lengths).tolist())
        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)

        # -set the hyperparameters
        model, model_info = bnpy.run(
            dataset,
            self.alloModel,
            self.obsModel,
            self.varMethod,
            nLap        = self.n_iteration,
            nTask       = self.nTask,
            nBatch      = self.nBatch,
            convergethr = self.convergethr,
            alpha       = self.alpha,
            gamma       = self.gamma,
            transAlpha  = self.transAlpha,
            startAlpha  = self.startAlpha,
            hmmKappa    = self.hmmKappa,
            sF          = self.sF,
            ECovMat     = self.ECovMat,
            K           = self.K,
            initname    = self.initname)

        self.model = model
        return self

    def score(self, X):
        if X.shape[0] == 1:
            X = np.append(X, X[0].reshape((1, -1)), axis=0)
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)
        log_probability = LP['evidence'] # by HongmiWu 28.07-2017
        return log_probability

    def calc_log(self, X):
        from scipy.misc import logsumexp
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)
        log = LP['E_log_soft_ev']
        log_curve = [logsumexp(log[i]) for i in range(len(log))]
        log_curve = np.cumsum(log_curve)
        return log_curve
