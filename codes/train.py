import os
import logging
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
logging.getLogger('tensorflow').disabled = True

# get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )
## change directory to the current directory where code is saved
os.chdir(dirname)

import tensorflow as tf
import numpy as np
import tqdm
from pathlib import Path
import sklearn.preprocessing as skp
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.logging.set_verbosity(tf.logging.ERROR)
tf.autograph.set_verbosity(0)
import model
import utils
import datasets, own_dataset


## mention paths
data_folder = os.path.join(os.path.dirname(dirname), 'data')
summaries = os.path.join(os.path.dirname(dirname), 'summaries')
output = os.path.join(os.path.dirname(dirname), 'output')
model_dir       = os.path.join(os.path.dirname(dirname), 'models')

## transformation task params
noise_param = 15 #noise_amount
scale_param = 1.1 #scaling_factor
permu_param = 20 #permutation_pieces
tw_piece_param = 9 #time_warping_pieces
twsf_param = 1.05 #time_warping_stretch_factor
no_of_task = ['original_signal', 'noised_signal', 'scaled_signal', 'negated_signal', 'flipped_signal', 'permuted_signal', 'time_warped_signal'] 
transform_task = [0, 1, 2, 3, 4, 5, 6] #transformation labels
single_batch_size = len(transform_task)
total_fold = 5
## hyper parameters
batchsize = 128  
actual_batch_size =  batchsize * single_batch_size
log_step = 100
epoch = 30
initial_learning_rate = 0.001
drop_rate = 0.6
regularizer = 1
L2 = 0.0001
lr_decay_steps = 10000
lr_decay_rate = 0.9
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
extract_data = 0
overlap_pct = 0
data_tag = 'mecg'
current_time    = utils.current_time()

print(current_time)

""" for the first time run this """ 
if extract_data == 1:
    if data_tag == 'aecg':
        martins_dataset.extract_martins_dataset_composite(overlap_pct=overlap_pct, window_size_sec=10, fs=256, data_path=data_folder, type_m_or_f=data_tag)
    else:        
        # martins_dataset.fetch_save_txt(type_m_or_f = data_tag)
        martins_dataset.extract_martins_dataset(overlap_pct=overlap_pct, window_size_sec=10, fs=256, data_path=data_folder, type_m_or_f=data_tag)

martins_data = np.load(os.path.join(data_folder, 'martins_'+ data_tag + '_' + str(overlap_pct)+'.npy'), allow_pickle=True)

graph = tf.Graph()
print('creating graph...')
with graph.as_default():
    
    ## initialize tensor
    
    input_tensor        = tf.compat.v1.placeholder(tf.float32, shape = (None, window_size, 1), name = "input")
    y                   = tf.compat.v1.placeholder(tf.float32, shape = (None, np.shape(transform_task)[0]), name = "output") 
    drop_out            = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="Drop_out")
    isTrain             = tf.placeholder(tf.bool, name = 'isTrain')
    global_step         = tf.Variable(0, dtype=np.float32, trainable=False, name="steps")

    conv1, conv2, conv3, main_branch, task_0, task_1, task_2, task_3, task_4, task_5, task_6 = model.self_supervised_model(input_tensor, isTraining= isTrain, drop_rate= drop_out)
    logits = [task_0, task_1, task_2, task_3, task_4, task_5, task_6]
    ## main branch is the output after all conv layers
    featureset_size = main_branch.get_shape()[1].value
    y_label = utils.get_label(y= y, actual_batch_size= actual_batch_size)
    all_loss = utils.calculate_loss(y_label, logits)
    output_loss = utils.get_weighted_loss(loss_coeff, all_loss)  
    
    if regularizer:
        l2_loss = 0
        weights = []
        for v in tf.trainable_variables():
            weights.append(v)
            if 'kernel' in v.name:
                l2_loss += tf.nn.l2_loss(v)
        output_loss = output_loss + l2_loss * L2
        
    y_pred                = utils.get_prediction(logits = logits)
    learning_rate         = tf.compat.v1.train.exponential_decay(initial_learning_rate, global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=True)

    optimizer             = tf.compat.v1.train.AdamOptimizer(learning_rate) 
    
    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        train_op    = optimizer.minimize(output_loss, global_step, colocate_gradients_with_ops=True)
        
    with tf.variable_scope('Session_saver'):
        saver       = tf.compat.v1.train.Saver(max_to_keep=150)

    tf.compat.v1.summary.scalar('learning_rate/lr', learning_rate)
    tf.compat.v1.summary.scalar('loss/training_batch_loss', output_loss)
    
    summary_op      = tf.compat.v1.summary.merge_all()    
        
print('graph creation finished')

""" Training """
# final_train_index, final_test_index = datasets.train_test_split_martin_kfold(data_folder, total_fold= total_fold, overlap_pct=overlap_pct, type_m_or_f=data_tag)

t = tqdm.trange(total_fold, desc='kfold', leave=True, ncols=100, bar_format='{l_bar}{bar}|')
for k in t:
   
    flag                    = k
    ## save STR results
    tr_ssl_result_filename  =  os.path.join(output, "STR_result"   , str("tr_" + str(k) +"_"  + current_time + ".npy"))
    te_ssl_result_filename  =  os.path.join(output, "STR_result"   , str("te_" + str(k) +"_"  + current_time + ".npy"))
    tr_ssl_loss_filename    =  os.path.join(output, "STR_loss"     , str("tr_" + str(k) +"_"  + current_time + ".npy"))
    te_ssl_loss_filename    =  os.path.join(output, "STR_loss"     , str("te_" + str(k) +"_"  + current_time + ".npy"))
    feature_saved_path      =  os.path.join(output , 'feature', 'fold_' + str(k) )
            
    str_logs        = os.path.join(summaries, "STR", current_time)
    er_logs         = os.path.join(summaries, "ER", current_time)
    utils.makedirs(str_logs)
    utils.makedirs(er_logs)
    utils.makedirs(feature_saved_path)
    
    martin_train_data, martin_test_data = datasets.train_test_split_martin_kfold(data_folder, kfold = k, total_fold= total_fold, overlap_pct=overlap_pct, type_m_or_f=data_tag)
    np.random.shuffle(martin_train_data)
    
    train_ECG   = martin_train_data[:,6:] 
    train_stress = tf.keras.utils.to_categorical(martin_train_data[:,1], 2)
    train_pss = martin_train_data[:,2]
    train_pdq = martin_train_data[:,3]
    train_fsi = martin_train_data[:,4]
    train_cortisol = martin_train_data[:,5]

    
    test_ECG   = martin_test_data[:,6:] 
    test_stress = tf.keras.utils.to_categorical(martin_test_data[:,1], 2)
    test_pss = martin_test_data[:,2]
    test_pdq = martin_test_data[:,3]
    test_fsi = martin_test_data[:,4]
    test_cortisol = martin_test_data[:,5]
    
    training_length = train_ECG.shape[0]
    testing_length  = test_ECG.shape[0]
    
    print('Initializing all parameters.')
    tf.reset_default_graph()
    with tf.Session(graph=graph) as sess:   
        summary_writer = tf.compat.v1.summary.FileWriter(str_logs, sess.graph)
    
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print('self supervised training started') 
        
        train_loss_dict = {}
        test_loss_dict  = {}
    
        tr_ssl_result = {}
        te_ssl_result = {}    
        
        ## epoch loop
        for epoch_counter in range(epoch):
            
            t.set_description("kfold - epoch: %i" %epoch_counter)
            t.refresh()
            
            tr_loss_task = np.zeros((len(transform_task), 1), dtype  = np.float32)
            train_pred_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            train_true_task = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32) -1
            tr_output_loss = 0
    
           
            tr_total_gen_op = utils.make_total_batch(data = train_ECG, length = training_length, batchsize = batchsize, 
                                               noise_amount=noise_param, 
                                               scaling_factor=scale_param, 
                                               permutation_pieces=permu_param, 
                                               time_warping_pieces=tw_piece_param, 
                                               time_warping_stretch_factor= twsf_param, 
                                               time_warping_squeeze_factor= 1/twsf_param)
    
            for training_batch, training_labels, tr_counter, tr_steps in tr_total_gen_op:
                
                ## run the model here 
                training_batch, training_labels = utils.unison_shuffled_copies(training_batch, training_labels)
                training_batch = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1)
                fetches = [all_loss, output_loss, y_pred, train_op]
                if tr_counter % log_step == 0:
                    fetches.append(summary_op)
                    
                fetched = sess.run(fetches, {input_tensor: training_batch, y: training_labels, drop_out: drop_rate, isTrain: True})
                
                if tr_counter % log_step == 0: # 
                    summary_writer.add_summary(fetched[-1], tr_counter)
                    summary_writer.flush()
    
                tr_loss_task = utils.fetch_all_loss(fetched[0], tr_loss_task) 
                tr_output_loss += fetched[1]
                
                train_pred_task = utils.fetch_pred_labels(fetched[2], train_pred_task)
                train_true_task = utils.fetch_true_labels(training_labels, train_true_task)

            ## loss after epoch
            tr_epoch_loss = np.true_divide(tr_loss_task, tr_steps)
            train_loss_dict.update({epoch_counter: tr_epoch_loss})
            tr_output_loss = np.true_divide(tr_output_loss, tr_steps)
            
            ## performance matrix after each epoch
            tr_epoch_accuracy, tr_epoch_f1_score = utils.get_results_ssl(train_true_task, np.asarray(train_pred_task, int))
            tr_ssl_result = utils.write_result(tr_epoch_accuracy, tr_epoch_f1_score, epoch_counter, tr_ssl_result)
            utils.write_summary(loss = tr_epoch_loss, total_loss = tr_output_loss, f1_score = tr_epoch_f1_score, epoch_counter = epoch_counter, isTraining = True, summary_writer = summary_writer)
            utils.write_result_csv(k, epoch_counter, os.path.join(output, "STR_result", "tr_str_f1_Score.csv"), tr_epoch_f1_score)
    
            model_path = os.path.join(model_dir , 'fold_' + str(k), "epoch_" + str(epoch_counter))
            utils.makedirs(model_path)
            save_path = saver.save(sess, os.path.join(model_path, "SSL_model.ckpt"))
            # print("Self-supervised trained model is saved in path: %s" % save_path) 
            
            ## initialize array
            te_loss_task    = np.zeros((len(transform_task), 1), dtype  = np.float32)
            test_pred_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            test_true_task  = np.zeros((len(transform_task), actual_batch_size), dtype  = np.float32)-1
            te_output_loss  = 0
           
            te_total_gen_op = utils.make_total_batch(data = test_ECG, 
                                                     length = testing_length, 
                                                     batchsize = batchsize, 
                                                     noise_amount=noise_param, 
                                                     scaling_factor=scale_param, 
                                                     permutation_pieces=permu_param, 
                                                     time_warping_pieces=tw_piece_param, 
                                                     time_warping_stretch_factor= twsf_param, 
                                                     time_warping_squeeze_factor= 1/twsf_param)
    
            for testing_batch, testing_labels, te_counter, te_steps in te_total_gen_op:
                
                ## run the model here 
                fetches = [all_loss, output_loss, y_pred]
                    
                fetched = sess.run(fetches, {input_tensor: testing_batch, y: testing_labels, drop_out: 0.0, isTrain: False})
    
                te_loss_task = utils.fetch_all_loss(fetched[0], te_loss_task)
                te_output_loss += fetched[1]
                test_pred_task = utils.fetch_pred_labels(fetched[2], test_pred_task)
                test_true_task = utils.fetch_true_labels(testing_labels, test_true_task)
    
            ## loss after epoch
            te_epoch_loss = np.true_divide(te_loss_task, te_steps)
            test_loss_dict.update({epoch_counter: te_epoch_loss})
            te_output_loss = np.true_divide(te_output_loss, te_steps)
    
            ## performance matrix after each epoch
            te_epoch_accuracy, te_epoch_f1_score = utils.get_results_ssl(test_true_task, test_pred_task)            
            te_ssl_result = utils.write_result(te_epoch_accuracy, te_epoch_f1_score, epoch_counter, te_ssl_result)    
            utils.write_summary(loss = te_epoch_loss, total_loss = te_output_loss, f1_score = te_epoch_f1_score, epoch_counter = epoch_counter, isTraining = False, summary_writer = summary_writer)
            utils.write_result_csv(k, epoch_counter, os.path.join(output, "STR_result", "te_str_f1_score.csv"), te_epoch_f1_score)
            
            x_tr_feature = utils.extract_feature(x_original = train_ECG, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
            x_te_feature = utils.extract_feature(x_original = test_ECG, featureset_size = featureset_size, batch_super = batchsize, input_tensor = input_tensor, isTrain = isTrain, drop_out = drop_out, extract_layer = main_branch, sess = sess)
            
            
            if epoch_counter==epoch-1:
                
                model.model_classification(x_tr_feature = x_tr_feature, y_tr = train_stress, x_te_feature = x_te_feature, y_te = test_stress, identifier = 'stress', kfold = flag, epoch_super = 100, result = output, summaries = er_logs, current_time = current_time)        
                model.model_regression(x_tr_feature = x_tr_feature, y_tr = train_pdq, x_te_feature = x_te_feature, y_te = test_pdq, identifier = 'pdq', kfold = flag, epoch_super = 100, result = output, summaries = er_logs, current_time = current_time)  
                model.model_regression(x_tr_feature = x_tr_feature, y_tr = train_pss, x_te_feature = x_te_feature, y_te = test_pss, identifier = 'pss', kfold = flag, epoch_super = 100, result = output, summaries = er_logs, current_time = current_time)  
                model.model_regression(x_tr_feature = x_tr_feature, y_tr = train_fsi, x_te_feature = x_te_feature, y_te = test_fsi, identifier = 'fsi', kfold = flag, epoch_super = 100, result = output, summaries = er_logs, current_time = current_time)  
                model.model_regression(x_tr_feature = x_tr_feature, y_tr = train_cortisol, x_te_feature = x_te_feature, y_te = test_cortisol, identifier = 'cortisol', kfold = flag, epoch_super = 100, result = output, summaries = er_logs, current_time = current_time)  
    

        np.save(tr_ssl_loss_filename, train_loss_dict)
        np.save(te_ssl_loss_filename, test_loss_dict)
    
        np.save(tr_ssl_result_filename, tr_ssl_result)
        np.save(te_ssl_result_filename, te_ssl_result)


