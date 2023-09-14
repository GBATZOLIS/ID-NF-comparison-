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

def main(config_path, save_path):
    # Read hyperparameters from the provided config_path
    config = parse_config(config_path)
    print(config.keys())
    n_sigmas = int(config['n_sigmas']) 
    datadim = int(config['data_dim'])
    batch_size = 100
    sig2_0 = float(config['sig2_min'])
    sig2_1 = float(config['sig2_max'])

    delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )
    sigmas = np.zeros(n_sigmas) + sig2_0
    for k in range(n_sigmas-1):
        sigmas[k+1] = sigmas[k] * np.exp(delta)

    sing_values_batch = np.abs(np.random.randn(n_sigmas, batch_size, datadim))

    # Check if save_path exists, create it if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    local_estimator = False
    data_type = 'vector'
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
            d_hat[k] = ID_NF_estimator(sing_values, sigmas, datadim, mode=data_type, latent_dim=10, plot=True, tag=str(k), save_path=save_path)
        print('--estimate mean ', d_hat.mean())
    else:
        if plot:
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            for d in range(datadim):
                sing_values_d = sing_values_batch[:,0,d]
                ax.plot(sigmas,sing_values_d) #,c=colors[n],label=labels[n]
            plt.yscale('log')#, nonposy='clip')
            plt.xscale('log')#, nonposx='clip')
            plt.savefig(os.path.join(save_path, 'sing_values_vs_sig2'+'.pdf'))

        d_hat = np.zeros(batch_size)
        for k in range(batch_size):
            sing_values = sing_values_batch[:,k,:]
            d_hat[k] = ID_NF_estimator(sing_values, sigmas, datadim, mode=data_type, latent_dim=10, plot=True, tag=str(k), save_path=save_path)
        
        print(d_hat)
        d_hat = d_hat.mean()
        print('--estimate mean ', d_hat.mean())

        
    np.save(os.path.join(save_path, 'd_hat.npy'), d_hat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate key quantity from saved singular values.")
    parser.add_argument("--config_path", type=str, default="config.txt", help="Path to the configuration file.")
    parser.add_argument("--save_path", type=str, default="outputs", help="Path where to save the output files.")

    args = parser.parse_args()
    
    main(args.config_path, args.save_path)

