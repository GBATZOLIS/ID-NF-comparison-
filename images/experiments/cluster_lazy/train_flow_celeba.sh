#!/bin/bash

#SBATCH --mail-user=<horvat@pyl.unibe.ch>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="celeba"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4 #4
#SBATCH --mem=32G  #32

#SBATCH --partition=gpu

#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:gtx1080ti:1

##SBATCH --gres=gpu:rtx3090:1

#SBATCH --array=1-3

cd /storage/homefs/ch19g182/Python/estimate_d/images/experiments

module load CUDA

######## CelebA-HQ #######
python train.py --resume 1 --epochs 300 --noise_type_preprocess non --scale_factor 1 --noise_type gaussian  --modelname july --dataset celeba --algorithm flow --outerlayers 20 --innerlayers 8 --levels 4 --linlayers 2 --linchannelfactor 1 --lineartransform lu --splinerange 10.0 --splinebins 11 --actnorm --batchsize 25 --lr 3.0e-4 --nllfactor 1 --uvl2reg 0.0 --clip 5.0 --validationsplit 0.1 --dropout 0.0 --dir /storage/homefs/ch19g182/Python/estimate_d
python evaluate_MNIST.py --OOD_dataset celeba --noise_type_preprocess non --scale_factor 1 --noise_type gaussian  --modelname july --dataset celeba --algorithm flow --outerlayers 20 --innerlayers 8 --levels 4 --linlayers 2 --linchannelfactor 1 --lineartransform lu --splinerange 10.0 --splinebins 11 --actnorm --evaluate 50 --estimate_d --dropout 0.0 --dir /storage/homefs/ch19g182/Python/estimate_d
