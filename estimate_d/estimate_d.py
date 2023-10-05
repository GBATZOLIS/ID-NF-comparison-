# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:48:13 2022
    Dummy code for estimate ID once singular values are calculated
@author: Horvat
"""
from matplotlib import pyplot as plt
import numpy as np
import os
from utils import ID_NF_estimator
import argparse
import configargparse

def parse_config(filename):
    config_dict = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the first line
            line = line.strip()
            if line.startswith('Num trainable params:'):
                break
            if ":" not in line:
                continue  # Skip lines without a colon
            key = line.split(':')[0].strip(" '")
            value = line.split(':', 1)[1].strip(" ,{}[]'\"")
            if value.isdigit():  # Convert numerical strings to integers
                value = int(value)
            elif '.' in value and value.replace('.', '', 1).isdigit():  # Convert float strings to floats
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            config_dict[key] = value

    return config_dict

def read_singular_values(read_base_path, n_sigmas, data_type='vector'):
    if data_type == 'vector':
        sing_values_batch = []
        for i in range(n_sigmas):
            folder_name = 'sig2_%d_seed_0' % i
            singular_values_path = os.path.join(read_base_path, folder_name, 'sing_values_%d.npy' % i)
            singular_values = np.load(singular_values_path)
            sing_values_batch.append(singular_values)
        sing_values_batch = np.stack(sing_values_batch)
        print(sing_values_batch.shape)
    elif data_type == 'image':
        sing_values_batch = []
        for i in range(n_sigmas):
            folder_name = 'sig2_%d' % i
            folder_path = os.path.join(read_base_path, folder_name)
            file_list = os.listdir(folder_path)

            # Filter the files to get only the .npy files
            npy_files = [file for file in file_list if file.endswith(".npy")]

            # Extract the indices from the filenames and find the maximum index
            max_index = max([int(file.split('_')[2].split('.')[0]) for file in npy_files])

            # Construct the filename for the numpy array with the highest index
            highest_index_filename = f"sing_values_{max_index}.npy"

            # Load the numpy array with the highest index
            singular_values = np.load(os.path.join(folder_path, highest_index_filename))
            sing_values_batch.append(singular_values)
        sing_values_batch = np.stack(sing_values_batch)
        print(sing_values_batch.shape)
    return sing_values_batch

def main(config_path, read_base_path, save_path):
    #config_path is given for vector data. For image data we use a configuration file
    data_type = 'vector' if config_path else 'image'

    # Read hyperparameters from the provided config_path
    if data_type == 'vector':
        config_path = os.path.join(read_base_path, 'sig2_0_seed_0', 'config.txt')
        config = parse_config(config_path)
        print(config.keys())
        n_sigmas = int(config['n_sigmas']) 
        datadim = int(config['data_dim'])
        #batch_size = int(config['ID_samples'])
        sig2_0 = float(config['sig2_min'])
        sig2_1 = float(config['sig2_max'])

        delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )
        sigmas = np.zeros(n_sigmas) + sig2_0
        for k in range(n_sigmas-1):
            sigmas[k+1] = sigmas[k] * np.exp(delta)
        
        sing_values_batch = read_singular_values(read_base_path, n_sigmas, data_type)

        batch_size = sing_values_batch.shape[1]

    
    elif data_type == 'image':
        sigmas = args.sigmas
        batch_size = args.ID_samples
        datadim = args.datadim
        sing_values_batch = read_singular_values(read_base_path, len(sigmas), data_type)
        batch_size = sing_values_batch.shape[1]
        
    # Check if save_path exists, create it if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    local_estimator = False
    plot = False
    if local_estimator:
        d_hat = np.zeros(batch_size)
        if plot:
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            for d in range(datadim):
                sing_values_d = sing_values_batch[:,0,d]
                ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]
            plt.yscale('log')#, nonposy='clip')
            plt.xscale('log')#, nonposx='clip')
            plt.savefig(os.path.join(save_path, 'sing_values_vs_sig2'+'.pdf'))

        for k in range(batch_size):
            sing_values = sing_values_batch[:,k,:]
            d_hat[k] = ID_NF_estimator(sing_values, sigmas, datadim, mode=data_type, latent_dim=args.latent_dim, plot=False, tag=str(k), save_path=save_path)
        print('--estimate mean ', d_hat.mean())
    else:
        if plot:
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            for s in range(len(sigmas)):
                ax.plot(np.sort(sing_values_batch[s,0,:])[::-1], label=sigmas[s])
            plt.legend()
            plt.yscale('log')#, nonposy='clip')
            ax.axvline(x=datadim-args.latent_dim, color='red')
            #plt.xscale('log')#, nonposx='clip')
            plt.savefig(os.path.join(save_path, 'sing_values_vs_sig2'+'.pdf'))

        d_hat = np.zeros(batch_size)
        for k in range(batch_size):
            sing_values = sing_values_batch[:,k,:]
            d_hat[k] = ID_NF_estimator(sing_values, sigmas, datadim, mode=data_type, latent_dim=args.latent_dim, plot=False, tag=str(k), save_path=save_path)
            print(d_hat[k])
        
        print(d_hat)
        d_hat = d_hat.mean()
        print('--estimate mean ', d_hat.mean())

    np.save(os.path.join(save_path, 'd_hat.npy'), d_hat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate key quantity from saved singular values.")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path", required=False)
    parser.add_argument("--config_path", type=str, default=None, help="Path to the configuration file.", required=False)
    parser.add_argument("--read_path", type=str, default='./', help="Base read path.")
    parser.add_argument("--save_path", type=str, default="outputs", help="Path where to save the output files.")

    # Define the arguments needed for ther loading of the image config
    parser.add_argument("--dataset", type=str, default="SquaresManifold", help="Dataset name")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--datadim", type=int, default=1024, help="Data dimensionality")
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent dimension")
    parser.add_argument("--split_ratio", type=float, default=0.88889, help="Split ratio")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--num_squares", type=int, default=10, help="Number of squares")
    parser.add_argument("--square_range", nargs="+", type=int, default=[3, 5], help="Square range")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[1e-09, 1e-1, 1], help="Sigmas")
    parser.add_argument("--ID_samples", type=int, default=10, help="ID samples")
    parser.add_argument("--algorithm", type=str, default="dnf", help="Algorithm")
    parser.add_argument("--modelname", type=str, default="paper", help="Model name")
    parser.add_argument("--run_on_gpu", action="store_true", default=True, help="Run on GPU")
    parser.add_argument("--multi_gpu", action="store_false", default=False, help="Use multiple GPUs")
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model latent dimension")
    parser.add_argument("--levels", type=int, default=4, help="Levels")
    parser.add_argument("--linlayers", type=int, default=2, help="Linear layers")
    parser.add_argument("--linchannelfactor", type=int, default=1, help="Linear channel factor")
    parser.add_argument("--outerlayers", type=int, default=20, help="Outer layers")
    parser.add_argument("--innerlayers", type=int, default=6, help="Inner layers")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Outer transform")
    parser.add_argument("--innertransform", type=str, default="rq-coupling", help="Inner transform")
    parser.add_argument("--lineartransform", type=str, default="lu", help="Linear transform")
    parser.add_argument("--splinerange", type=float, default=10.0, help="Spline range")
    parser.add_argument("--splinebins", type=int, default=11, help="Spline bins")
    parser.add_argument("--sig2", type=float, default=0.01, help="Sig2")
    parser.add_argument("--actnorm", action="store_true", default=True, help="Activate actnorm")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--batchsize", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="Learning rate")
    parser.add_argument("--l1", action="store_false", default=False, help="Use L1 loss")
    parser.add_argument("--msefactor", type=float, default=1000.0, help="MSE factor")
    parser.add_argument("--nllfactor", type=float, default=1.0, help="NLL factor")
    parser.add_argument("--uvl2reg", type=float, default=0.01, help="UVL2 regularization")
    parser.add_argument("--weightdecay", type=float, default=1.0e-5, help="Weight decay")
    parser.add_argument("--validationsplit", type=float, default=0.1, help="Validation split")
    parser.add_argument("--clip", type=float, default=5.0, help="Clip value")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--debug", action="store_false", default=False, help="Debug mode")
    parser.add_argument("--dir", type=str, default="/store/CIA/gb511/projects/dim_estimation/experiments/ID-NF/SquaresManifold", help="Directory path")

    args = parser.parse_args()
    
    main(args.config_path, args.read_path, args.save_path)

