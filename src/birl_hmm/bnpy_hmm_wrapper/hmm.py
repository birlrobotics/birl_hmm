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
        nTask = 5,                 # Number of runs to perform for single experiment
        nBatch = 1,                # Number of batches (aka minibatches) to split up dataset into.
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
            initname    = self.initname,
            taskID      = 1,
            output_path = '/tmp/kitting_experiments/',
            )
        self.log_startprob = np.log(model.allocModel.get_init_prob_vector())
        self.log_transmat  = np.log(model.allocModel.get_trans_prob_matrix())
        self.model = model
        return self

    def score(self, X):
        '''
        Compute the the joint log-likelihood p(x_1, x_2,....,x_T)
        '''
        
        if X.shape[0] == 1:
            X = np.append(X, X[0].reshape((1, -1)), axis=0)
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        resp, respPair, logMargPrSeq = bnpy.allocmodel.hmm.HMMUtil.FwdBwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), logSoftEv)
        return logMargPrSeq
        

    def calc_log(self, X):
        from scipy.misc import logsumexp
        Xprev  = X[:-1,:]
        X      = X[1:,:]
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)

        logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        SoftEv, lognormC = bnpy.allocmodel.hmm.HMMUtil.expLogLik(logSoftEv)
        fmsg, margPrObs = bnpy.allocmodel.hmm.HMMUtil.FwdAlg(np.exp(self.log_startprob), np.exp(self.log_transmat), SoftEv)
        log_curve = np.log(margPrObs) + lognormC
        log_curve = np.cumsum(log_curve)
        return log_curve

    def predict(self, X, lengths=None):
        Xprev      = X[:-1,:]
        X          = X[1:,:]
        doc_range  = list([0])
        lengths = [X.shape[0]]
        doc_range += (np.cumsum(lengths).tolist())
        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)        
        for n in range(dataset.nDoc):
            logSoftEv = self.model.obsModel.calcLogSoftEvMatrix_FromPost(dataset.make_subset([n]))
            SoftEv, lognormC = bnpy.allocmodel.hmm.HMMUtil.expLogLik(logSoftEv)            
            zHat =  self.runViterbiAlg(logSoftEv, self.log_startprob, self.log_transmat)
        return zHat

    def runViterbiAlg(self, logSoftEv, logPi0, logPi):
        
        ''' Run viterbi algorithm to estimate MAP states for single sequence.
        Args
        ------
        logSoftEv : 2D array, T x K
            log soft evidence matrix
            each row t := log p( x[t] | z[t]=k )
        pi0 : 1D array, length K
            initial state probability vector, sums to one
        pi : 2D array, shape K x K
            j-th row is is transition probability vector for state j
        Returns
        ------
        zHat : 1D array, length T, representing the MAP state sequence
            zHat[t] gives the integer label {1, 2, ... K} of state at timestep t
        '''

        from bnpy.util import EPS
        
        if np.any(logPi0 > 0):
            logPi0 = np.log(logPi0 + EPS)
        if np.any(logPi > 0):
            logPi = np.log(logPi + EPS)
        T, K = np.shape(logSoftEv)

        # ScoreTable : 2D array, shape T x K
        #   entry t,k gives the log probability of reaching state k at time t
        #   under the most likely path
        ScoreTable = np.zeros((T, K))

        # PtrTable : 2D array, size T x K
        #   entry t,k gives the integer id of the state j at timestep t-1
        #   which would be part of the most likely path to reaching k at t
        PtrTable = np.zeros((T, K))

        ScoreTable[0, :] = logSoftEv[0] + logPi0
        PtrTable[0, :] = -1

        ids0toK = range(K)
        for t in xrange(1, T):
            ScoreMat_t = logPi + ScoreTable[t - 1, :][:, np.newaxis]
            bestIDvec = np.argmax(ScoreMat_t, axis=0)

            PtrTable[t, :] = bestIDvec
            ScoreTable[t, :] = logSoftEv[t, :] \
                + ScoreMat_t[(bestIDvec, ids0toK)]

        # Follow backward pointers to construct most likely state sequence
        z = np.zeros(T)
        z[-1] = np.argmax(ScoreTable[-1])
        for t in reversed(xrange(T - 1)):
            z[t] = PtrTable[int(t + 1), int(z[t + 1])]
        return z
    
