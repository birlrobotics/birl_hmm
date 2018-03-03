import numpy as np
import copy
from parameters_combinator import ParametersCombinator
import ipdb

def model_factory(model_type, model_config):
    if model_type == 'hmmlearn\'s HMM':
        import hmmlearn.hmm 
        model = hmmlearn.hmm.GaussianHMM(
            params="mct", 
            init_params="cmt", 
            **model_config
        )
        n_components = model.n_components
        start_prob = np.zeros(n_components)
        start_prob[0] = 1
        model.startprob_ = start_prob
        return model
    elif model_type == 'BNPY\'s HMM':
        import birl_hmm.bnpy_hmm_wrapper.hmm
        model = birl_hmm.bnpy_hmm_wrapper.hmm.HongminHMM(**model_config)
        return model

def get_model_generator(model_type, model_config):
    pc = ParametersCombinator(model_config)
    for i in pc.iteritems():
        yield model_factory(model_type, i), i
