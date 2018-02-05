import bnpy
import numpy as np

class HongminHMM():
    def __init__(
        self, 
        alloModel,
        obsModel,
        varMethod,
        n_iteration,
        K,
        nTask = 1,
        nBatch = 10,
        convergethr = 0.000000001, #for memoVB
        alpha = 0.5,
        gamma = 5.0,
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
        self.sF = sF
        self.ECovMat = ECovMat
        self.K = K
        self.initname = initname

    def fit(self, X, lengths):
        Xprev      = X

        doc_range  = list([0])
        doc_range += (np.cumsum(lengths).tolist())

        dataset    = bnpy.data.GroupXData(X, doc_range, None, Xprev)

        # -set the hyperparameters
        model, model_info = bnpy.run(
            dataset,
            self.alloModel,
            self.obsModel,
            self.varMethod,
            #output_path = os.path.join(model_save_path, 'results'),
            nLap = self.n_iteration,
            nTask = self.nTask,
            nBatch = self.nBatch,
            convergethr = self.convergethr,
            alpha = self.alpha,
            gamma = self.gamma,
            sF = self.sF,
            ECovMat = self.ECovMat,
            K = self.K,
            initname = self.initname)

        self.model = model
        return self

    def score(self, X):
        Xprev = X
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)

        #SS = self.model.get_global_suff_stats(dataset, LP)
        #log_probability = self.model.obsModel.calcMargLik(SS)
        
        log_probability = LP['evidence'] # by HongmiWu 28.07-2017

        return log_probability

    def calc_log(self, X):
        from scipy.misc import logsumexp
        import ipdb
        Xprev = X
        length = len(X)
        doc_range = [0, length]
        dataset = bnpy.data.GroupXData(X, doc_range, length, Xprev)
        LP = self.model.calc_local_params(dataset)

        log = LP['logLik_n']
        log_curve = [logsumexp(log[i]) for i in range(len(log))]
        log_curve = np.cumsum(log_curve)

        #SS = self.model.get_global_suff_stats(dataset, LP)
        #log_probability = self.model.obsModel.calcMargLik(SS)

        return log_curve
