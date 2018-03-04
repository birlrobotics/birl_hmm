import numpy as np
import model_generation
import model_score
import ipdb

def run(
    list_of_train_mat,
    list_of_test_mat,
    model_type,
    model_config,
    score_metric,
):
    train_lengths = [i.shape[0] for i in list_of_train_mat]
    train_lengths[-1] -= 1 #for autoregressive observation
    train_X = np.concatenate(list_of_train_mat, axis=0)
    test_lengths = [i.shape[0] for i in list_of_test_mat]
    test_X = np.concatenate(list_of_test_mat, axis=0)

    model_list = []
    model_generator = model_generation.get_model_generator(model_type, model_config)
    for model, now_model_config in model_generator:
        print
        print '-'*20
        print ' working on config:', now_model_config

        try:
            model = model.fit(train_X, lengths=train_lengths)
            score = model_score.score(score_metric, model, test_X, test_lengths)
        except Exception as e:
            print "Failed to train this model, will ignore it: %s"%e
            continue
            
        if score == None:
            print "scorer says to skip this model, will do"
            continue

        model_list.append({
            "model": model,
            "now_model_config": now_model_config,
            "score": score
        })
        print 'score:', score 
        print '='*20
        print 

    sorted_model_list = sorted(model_list, key=lambda x:x['score'])

    if len(sorted_model_list) == 0:
        print "cannot train model for state %s"%(state_no,)
        return None

    return sorted_model_list
