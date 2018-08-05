import numpy as np
import model_generation
import model_score
from sklearn.model_selection import KFold
import copy
import sys, traceback
import coloredlogs, logging
coloredlogs.install()
import ipdb

def run(
    list_of_train_mat,
    list_of_test_mat,
    model_type,
    model_config,
    score_metric,
    logger=None
):
    if logger is None:
        logger = logging.getLogger('birl_hmm_train_model')

    list_of_train_mat = np.array(list_of_train_mat)
    list_of_test_mat = np.array(list_of_test_mat)

    tried_models = []
    model_generator = model_generation.get_model_generator(model_type, model_config)
    for raw_model, model_config in model_generator:
        logger.debug('-'*20)
        logger.debug(' working on config: %s'%model_config)

        try:
            kf = KFold(n_splits=3, shuffle=True)
            scores = []
            for cv_train_index, cv_test_index in kf.split(list_of_train_mat):
                list_of_cv_train_mat = (list_of_train_mat.copy())[cv_train_index]
                list_of_cv_test_mat = (list_of_train_mat.copy())[cv_test_index]
                cv_train_lengths = [i.shape[0] for i in list_of_cv_train_mat]
                cv_train_lengths[-1] -= 1 #for autoregressive observation
                cv_train_X = np.concatenate(list_of_cv_train_mat, axis=0)
                cv_test_lengths = [i.shape[0] for i in list_of_cv_test_mat]
                cv_test_X = np.concatenate(list_of_cv_test_mat, axis=0)

                model = model_generation.model_factory(model_type, model_config)
                model = model.fit(cv_train_X, lengths=cv_train_lengths)
                score = model_score.score(score_metric, model, cv_test_X, cv_test_lengths)
                    
                if score == None:
                    raise Exception("scorer says to skip this model")
                else:
                    scores.append(score)
        except Exception as e:
            logger.error("Failed to run CV on this model: %s"%e)
            logger.error("traceback: %s"%traceback.format_exc())
            continue

        tried_models.append({
            "model": model,
            "model_config": model_config,
            "cv_score_mean": np.mean(scores),
            "cv_score_std": np.std(scores),
        })
        logger.debug('score: %s'%score)
        logger.debug('='*20)

    if len(tried_models) == 0:
        raise Exception("All models tried failed to train.")
    tried_models = sorted(tried_models, key=lambda x:x['cv_score_mean'])
    best_model = tried_models[0]['model'] 
    test_score = tried_models[0]['cv_score_mean']
    return best_model, test_score, tried_models
