# Import statements
from __future__ import print_function

import sys
sys.setrecursionlimit(10000)

from state_of_art_cnns.densenet import densenet

from dl_utilities.trip_loss import trip_utils
from dl_utilities.snapshots import snapshot_train_utils as snap_train
from dl_utilities.snapshots import snapshot_eval_utils as snap_eval
from dl_utilities.datasets import dataset_utils as ds_utils
from dl_utilities.callbacks import callback_utils as cb_utils

import numpy as np
import tensorflow as tf
import cPickle as pickle
import itertools, os, math

from shutil import copyfile

from keras import metrics
from keras import backend as K
from keras import utils as k_utils

from keras.optimizers import *
from keras.models import Model
from keras.datasets import cifar100
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Activation



# Global training variables
weight_dirname = "./snap_weights/"
history_dirname = "./snap_histories/"
eval_dirname = "./snap_final_evals/"


trip_worker = None
legacy_preprocessing = False
valid_arguments = [ "normal_model", "optimize_snap" ]

nb_classes = 100
batch_size = 24

initial_sgd_lr = 0.125
total_nb_epoch = 155



# DN model specifically for Snapshot training (in main routine below)
def get_scratch_dn100_model():
    return densenet.DenseNet((32, 32, 3), depth=100,
                    nb_dense_block=3, nb_layers_per_block=[20, 42, 34],
                    bottleneck=True, reduction=0.4, growth_rate=20,
                    weights=None, dropout_rate=0.0, classes=nb_classes)

                    
                         
                         
###################################  Main Functionality   ###################################

if __name__ == '__main__':
    # Do sanity checks of environment
    arg_string = ", ".join(valid_arguments)
    assert len(sys.argv) == 2, ("Must supply exactly 2 arguments.\n" + \
                                "Valid options are:  " + arg_string)

    assert sys.argv[1] in valid_arguments, ("Must supply exactly 2 arguments.\n" + \
                                "Valid options are:  " + arg_string)

                                
    if not os.path.isdir(weight_dirname):
        os.mkdir(weight_dirname)

    if not os.path.isdir(history_dirname):
        os.mkdir(history_dirname)

    if not os.path.isdir(eval_dirname):
        os.mkdir(eval_dirname)


    # Start up triplet training threads
    # Must be called before set_session or any other TF ops:
    #	https://devtalk.nvidia.com/default/topic/973477/cuda-programming-and-performance/-cuda8-0-bug-child-process-forked-after-cuinit-get-cuda_error_not_initialized-on-cuinit-/
    trip_worker = trip_utils.multi_thread_trip_gen(queue_size=32, nthreads=10)


    # Set TF session
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_cores = 6
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores)

    # https://github.com/tensorflow/tensorflow/blob/30b52579f6d66071ac7cdc7179e2c4aae3c9cb88/tensorflow/core/protobuf/config.proto#L35
    # If true, the allocator does not pre-allocate the entire specified
    # GPU memory region, instead starting small and growing as needed.
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)
    K.set_session(sess)


    # Get training and test data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    #  Normalize and kind of standardize data
    if legacy_preprocessing:
        x_train, x_test = ds_utils.simple_image_preprocess(x_train, x_test)
    else:
        x_train, x_test = ds_utils.normal_image_preprocess(x_train, x_test)            

    # Convert class vectors to sparse/binary class matrices
    y_train = k_utils.to_categorical(y_train, nb_classes)
    y_test = k_utils.to_categorical(y_test, nb_classes)

    quit()
    
    # Set up image augmentation generator
    global_image_aug = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=(4. / x_train.shape[2]), 
        height_shift_range=(4. / x_train.shape[1]), 
        horizontal_flip=True, 
        zoom_range=0.15)

    if (sys.argv[1] == "optimize_snap"):
        # List sequences and other hyper-parameters					
        seqs = [ [snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT, 
                        snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT], 
                    [snap_train.COSINE_SGD_OPT, snap_train.TRIP_L2_LOSS, snap_train.COSINE_SGD_OPT],
                    [snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT], 
                    [snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT],
                    [snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT,
                        snap_train.COSINE_SGD_OPT, snap_train.COSINE_SGD_OPT] ]
                    
        init_lrs = [ [ 0.2, 0.2, 0.2, 0.2, 0.2 ], 
                        [ 0.2, 0.225, 0.2 ],
                        [ 0.2, 0.2, 0.2 ],
                        [ 0.2, 0.2, 0.2 ],
                        [0.2, 0.225, 0.25, 0.275, .3] ]

        iters = [ [ 36, 30, 30, 30, 30 ],
                    [ 45, 66, 45 ],
                    [ 52, 52, 52 ],
                    [ 52, 52, 52 ],
                    [ 20, 40, 36, 32, 28 ] ]

        restart_indices = [ None,
                            [2],
                            None,
                            [1, 2],
                            None ]   
                            
        
        # Begin training of every listed sequence
        kwargs = { "trip_worker": trip_worker }
        seq_component_list = [ (seq, lr, epoch, index) 
                                for seq, lr, epoch, index in 
                                zip(seqs, init_lrs, iters, restart_indices) ]
        
        snap_train.conduct_snapshot_training(seq_component_list, (x_train, y_train), 
                                                (x_test, y_test), batch_size,
                                                get_scratch_dn100_model, weight_dirname, 
                                                history_dirname, global_image_aug, 
                                                **kwargs)
                                                 
        # Evaluate snapshot ensembles with basic and advanced metrics
        snap_eval.save_basic_ensemble_evals(seq_component_list, 
                                    (x_test, y_test), get_scratch_dn100_model,
                                    weight_dirname, eval_dirname)
                
        snap_eval.save_advanced_ensemble_evals(seq_component_list, 
                                    (x_test, y_test), get_scratch_dn100_model,
                                    weight_dirname, eval_dirname)
        
        
    # Otherwise, train a single normal model over the same total time period
    else:
        model = get_scratch_dn100_model()
        
        callbacks = [ ModelCheckpoint(weight_dirname + "normal_model.hdf5", 
                            monitor="acc", period=5,
                            save_best_only=False, save_weights_only=False) ]
        
        opt = SGD(lr=std_sgd_lr)
        callbacks.append(cb_utils.CosineLRScheduler(initial_sgd_lr, total_nb_epoch))

        snap_train.compile_and_train_single_model(
                                    model, opt, (x_train, y_train), 
                                    (x_test, y_test), total_nb_epoch, 
                                    history_filepath=(history_dirname + "normal_model.pickle"),
                                    aug_gen=global_image_aug, callbacks=callbacks)		

                                    
    # Stop triplet worker threads
    trip_worker.stop_all_threads()

    
    # Exit successfully	
    sys.exit(0)
