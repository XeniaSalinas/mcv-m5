#!/usr/bin/env python
import argparse
import os
import time
import sys
import numpy as np
from getpass import getuser
import matplotlib
matplotlib.use('Agg')  # Faster plot

# Import tools
from config.configuration import Configuration
from tools.logger import Logger
from tools.dataset_generators import Dataset_Generators
from tools.optimizer_factory import Optimizer_Factory
from callbacks.callbacks_factory import Callbacks_Factory
from models.model_factory import Model_Factory

from numpy  import array

# Train the network
def process(cf):
    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Create the data generators
    train_gen, valid_gen, test_gen = Dataset_Generators().make(cf)

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = Optimizer_Factory().make(cf)

    # Build model
    print ('\n > Building model...')
    model = Model_Factory().make(cf, optimizer)

    # Build the second model for ensembling
    if hasattr(cf, 'model_name_2'):
        if cf.model_name_2 != None:
            model2 = Model_Factory().make(cf, optimizer, model_name=cf.model_name_2)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf, valid_gen)

    if cf.train_model:
        # Train the model
        model.train(train_gen, valid_gen, cb)

    if cf.test_model:

        # Check if a second model is selected to apply an Ensemble of models
        if hasattr(cf, 'model_name_2'):
            if cf.model_name_2 != None:
                print('\n > Testing the model using an ensemble of models...')
                # Compute test metrics
                start_time_global = time.time()

                #Test first model
                test_metrics_model, metrics_names = model.test(test_gen, model_ensemble=True)
                test_metrics_model = array(test_metrics_model)

                #Test second model
                weights_test_file_model2 = os.path.join(cf.savepath, cf.weights_file_2)
                test_metrics_model_2, metrics_names = model2.test(test_gen, model_ensemble=True, weights_file_2=weights_test_file_model2)
                test_metrics_model_2 = array(test_metrics_model_2)

                A1 = 1.5
                A2 = 1
                # Perform the mean of the metrics
                total_metrics = A1*test_metrics_model + A2*test_metrics_model_2 / 2

                total_metrics.tolist()

                total_time_global = time.time() - start_time_global
                fps = float(cf.dataset.n_images_test) / total_time_global
                s_p_f = total_time_global / float(cf.dataset.n_images_test)
                print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time_global, fps, s_p_f))
                metrics_dict = dict(zip(metrics_names, total_metrics))
                print ('   Test metrics: ')
                for k in metrics_dict.keys():
                    print ('      {}: {}'.format(k, metrics_dict[k]))

                if cf.problem_type == 'segmentation':
                    # Compute Jaccard per class
                    metrics_dict = dict(zip(metrics_names, total_metrics))
                    I = np.zeros(cf.dataset.n_classes)
                    U = np.zeros(cf.dataset.n_classes)
                    jacc_percl = np.zeros(cf.dataset.n_classes)
                    for i in range(cf.dataset.n_classes):
                        I[i] = metrics_dict['I' + str(i)]
                        U[i] = metrics_dict['U' + str(i)]
                        jacc_percl[i] = I[i] / U[i]
                        print ('   {:2d} ({:^15}): Jacc: {:6.2f}'.format(i,
                                                                         cf.dataset.classes[i],
                                                                         jacc_percl[i] * 100))
                    # Compute jaccard mean
                    jacc_mean = np.nanmean(jacc_percl)
                    print ('   Jaccard mean: {}'.format(jacc_mean))
            else:
                # Compute test metrics
                model.test(test_gen)
        else:
            # Compute test metrics
            model.test(test_gen)


    if cf.pred_model:
        # Compute test metrics
        model.predict(test_gen, tag='pred')

    # Finish
    print (' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Sets the backend and GPU device.
class Environment():
    def __init__(self, backend='tensorflow'):
        #backend = 'tensorflow' # 'theano' or 'tensorflow'
        os.environ['KERAS_BACKEND'] = backend
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # "" to run in CPU, extra slow! just for debuging
        if backend == 'theano':
            # os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'
            """ fast_compile que lo que hace es desactivar las optimizaciones => mas lento """
            os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.95'
            print('Backend is Theano now')
        else:
            print('Backend is Tensorflow now')


# Main function
def main():
    # Define environment variables
    # Environment()

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default=None, help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default='/home/master/m5_project/data/', help='Name of the experiment')
    parser.add_argument('-l', '--local_path', type=str,
                        default='/home/master/m5_project/data/', help='Name of the experiment')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/pathname'\
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the '\
                                           'experiment using -e name in the '\
                                           'command line'

    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path, 'Datasets')
    experiments_path = os.path.join(local_path, getuser(), 'Experiments')
    shared_experiments_path = os.path.join(shared_path, getuser(), 'Experiments')
    usr_path = os.path.join('/home/', getuser())

    # Load configuration files
    configuration = Configuration(arguments.config_path, arguments.exp_name,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path)
    cf = configuration.load()

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # Copy result to shared directory
    configuration.copy_to_shared()


# Entry point of the script
if __name__ == "__main__":
    main()
