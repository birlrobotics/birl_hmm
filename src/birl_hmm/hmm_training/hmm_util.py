import numpy as np

def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

def fast_log_curve_calculation(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])
        logprobij, _fwdlattice = model._do_forward_pass(framelogprob)

        log_curve = [logsumexp(_fwdlattice[i]) for i in range(len(_fwdlattice))]

        return log_curve 
    elif issubclass(type(model.model), bnpy.HModel):
        return model.calc_log(X)
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

def get_emission_log_prob_matrix(X, model):
    import hmmlearn.hmm
    import hongminhmmpkg.hmm
    import bnpy

    if issubclass(type(model), hmmlearn.hmm._BaseHMM):
        from sklearn.utils import check_array, check_random_state
        from scipy.misc import logsumexp

        X = check_array(X)

        framelogprob = model._compute_log_likelihood(X[:])

        return framelogprob 
    elif issubclass(type(model.model), bnpy.HModel):
        raise Exception('hongmin BNPY not supported for now.')
    else:
        raise Exception('model of type %s is not supported by fast_log_curve_calculation.'%(type(model),))

