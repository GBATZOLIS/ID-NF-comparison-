#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:30:45 2022

@author: chrvt
"""

import logging
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time
import os
import random

import matplotlib.pyplot as plt
import io
import PIL
from torchvision import transforms

logger = logging.getLogger(__name__)

def plot_spectrum(singular_values, return_tensor=False, title='Spectrum', ground_truth=None, yaxis='normal'):    
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(15,10))
    plt.grid(alpha=0.5)
    plt.title(title)
    plt.xticks(np.arange(0, len(singular_values[0])+1, 10))

    if yaxis == 'log':
        plt.yscale('log')  # Set the y-axis to logarithmic scale

    if ground_truth:
        if isinstance(ground_truth, list):
            for gt in ground_truth:
                plt.axvline(x=len(singular_values[0])-gt, color='red', ls='--')
        else:
            plt.axvline(x=len(singular_values[0])-ground_truth, color='red', ls='--')

    for sing_vals in singular_values:
        #plt.bar(list(range(1, len(sing_vals)+1)),sing_vals)
        plt.plot(list(range(1, len(sing_vals)+1)), sing_vals)

    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image
    else:
        return plt.gcf()


class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class BaseTrainer(object):
    """ Training functionality shared between normal trainers and alternating trainers. """

    def __init__(self, model, run_on_gpu=True, multi_gpu=True, double_precision=False, device=0):
        self.model = model

        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.multi_gpu = self.run_on_gpu and multi_gpu and torch.cuda.device_count() > 1

        self.device = torch.device("cuda:%d" % device if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float
        if self.run_on_gpu and double_precision:
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
        elif self.run_on_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        elif double_precision:
            torch.set_default_tensor_type("torch.DoubleTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.model = self.model.to(self.device, self.dtype)
        # Verify everything is on the same device
        self.model.outer_transform = self.model.outer_transform.to(self.device)
        self.model.inner_transform = self.model.inner_transform.to(self.device)

        self.last_batch = None

        logger.info(
            "Training on %s with %s precision",
            "{} GPUS".format(torch.cuda.device_count()) if self.multi_gpu else "GPU" if self.run_on_gpu else "CPU",
            "double" if double_precision else "single",
        )

    def check_early_stopping(self, best_loss, best_model, best_epoch, loss, i_epoch, early_stopping_patience=None):
        try:
            loss_ = loss[0]
        except:
            loss_ = loss

        if best_loss is None or loss_ < best_loss:
            best_loss = loss_
            best_model = self.model.state_dict()
            best_epoch = i_epoch

        if early_stopping_patience is not None and i_epoch - best_epoch > early_stopping_patience >= 0:
            raise EarlyStoppingException

        return best_loss, best_model, best_epoch

    def wrap_up_early_stopping(self, best_model, currrent_loss, best_loss, best_epoch):
        try:
            loss_ = currrent_loss[0]
        except:
            loss_ = currrent_loss

        if loss_ is None or best_loss is None:
            logger.warning("Loss is None, cannot wrap up early stopping")
        elif best_loss < loss_:
            logger.info("Early stopping after epoch %s, with loss %8.5f compared to final loss %8.5f", best_epoch + 1, best_loss, loss_)
            self.model.load_state_dict(best_model)
        else:
            logger.info("Early stopping did not improve performance")

    @staticmethod
    def report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
        logging_fn = logger.info if verbose else logger.debug

        def contribution_summary(labels, contributions):
            summary = ""
            for i, (label, value) in enumerate(zip(labels, contributions)):
                if i > 0:
                    summary += ", "
                summary += "{}: {:>6.3f}".format(label, value)
            return summary

        try:
            train_report = "Epoch {:>3d}: train loss {:>8.5f} +/- {:>8.5f} ({})".format(
                i_epoch + 1, loss_train[0], loss_train[1], contribution_summary(loss_labels, loss_contributions_train)
            )
        except:
            train_report = "Epoch {:>3d}: train loss {:>8.5f} ({})".format(i_epoch + 1, loss_train, contribution_summary(loss_labels, loss_contributions_train))
        logging_fn(train_report)

        if loss_val is not None:
            try:
                val_report = "           val. loss  {:>8.5f} +/- {:>8.5f} ({})".format(loss_val[0], loss_val[1], contribution_summary(loss_labels, loss_contributions_val))
            except:
                val_report = "           val. loss  {:>8.5f} ({})".format(loss_val, contribution_summary(loss_labels, loss_contributions_val))
            logging_fn(val_report)

    @staticmethod
    def _check_for_nans(label, *tensors, fix_until=None, replace=0.0):
        for tensor in tensors:
            if tensor is None:
                continue

            if torch.isnan(tensor).any():
                n_nans = torch.sum(torch.isnan(tensor)).item()
                if fix_until is not None:
                    if n_nans <= fix_until:
                        logger.debug("%s contains %s NaNs, setting them to zero", label, n_nans)
                        tensor[torch.isnan(tensor)] = replace
                        return

                logger.warning("%s contains %s NaNs, aborting training!", label, n_nans)
                raise NanException

    @staticmethod
    def make_dataloader(dataset, validation_split, batch_size, n_workers=4):
        logger.debug("Setting up dataloaders with %s workers", n_workers)

        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

            n_samples = len(dataset)
            print('n_samples: %d' % n_samples)
            indices = list(range(100,n_samples))  #makes sure that first 100 images are always in traning set
            split = int(np.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = list(range(0,100)) + indices[split:] , indices[:split]  #makes sure that first 100 images are always in traning set

            # np.save(create_filename("sample", "validation_index", args), valid_idx) #sanity check
            
            print("Training partition indices: %s...", train_idx[:10])
            print("Validation partition indices: %s...", valid_idx[:10])
            logger.debug("Training partition indices: %s...", train_idx[:10])
            logger.debug("Validation partition indices: %s...", valid_idx[:10])

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                # pin_memory=self.run_on_gpu,
                num_workers=n_workers,
            )
            #val_loader = None
            val_loader = DataLoader(
                 dataset,
                 sampler=val_sampler,
                 batch_size=batch_size,
                 # pin_memory=self.run_on_gpu,
                 num_workers=n_workers,
            )
            
        return train_loader, val_loader

    @staticmethod
    def sum_losses(contributions, weights):
        loss = weights[0] * contributions[0]
        for _w, _l in zip(weights[1:], contributions[1:]):
            loss = loss + _w * _l
        return loss

    def optimizer_step(self, optimizer, loss, clip_gradient, parameters):
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(parameters, clip_gradient)
            # grad_norm = clip_grad_norm_(parameters, clip_gradient)
            # logger.debug("  Gradient norm (clipping at %s): %s", clip_gradient, grad_norm)
        optimizer.step()

    @staticmethod
    def _set_verbosity(epochs, verbose):
        # Verbosity
        if verbose == "all":  # Print output after every epoch
            n_epochs_verbose = 1
        elif verbose == "many":  # Print output after 2%, 4%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 50, 0)), 1)
        elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 20, 0)), 1)
        elif verbose == "few":  # Print output after 20%, 40%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 5, 0)), 1)
        elif verbose == "none":  # Never print output
            n_epochs_verbose = epochs + 2
        else:
            raise ValueError("Unknown value %s for keyword verbose", verbose)
        return n_epochs_verbose



class Trainer(BaseTrainer):
    """ Base trainer class. Any subclass has to implement the forward_pass() function. """

    def train(
        self,
        dataset,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.AdamW,
        optimizer_kwargs=None,
        initial_lr=1.0e-3,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs=None,
        restart_scheduler=None,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=1.0,
        verbose="all",
        parameters=None,
        callbacks=None,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
        seed=None,
        initial_epoch=None,
        sig2 = 0.0,
        noise_type = None,
        writer=None,
        save_dir=None,
        latent_dim=None
    ):

        if initial_epoch is not None and initial_epoch >= epochs:
            logging.info("Initial epoch is larger than epochs, nothing to do in this training phase!")
        elif initial_epoch is not None and initial_epoch <= 0:
            initial_epoch = None

        if loss_labels is None:
            loss_labels = [fn.__name__ for fn in loss_functions]
            
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        """    
        seed = 1237
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        """
        logger.debug("Initialising training data")
        train_loader, val_loader = self.make_dataloader(dataset, validation_split, batch_size)
        
        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        if parameters is None:
            parameters = list(self.model.parameters())
        opt = optimizer(parameters, lr=initial_lr, **optimizer_kwargs)
        
        
        logger.debug("Setting up LR scheduler")
        if epochs < 2:
            scheduler = None
            logger.info("Deactivating scheduler for only %s epoch", epochs)
        scheduler_kwargs = {} if scheduler_kwargs is None else scheduler_kwargs
        sched = None
        epochs_per_scheduler = restart_scheduler if restart_scheduler is not None else epochs
        if scheduler is not None:
            try:
                sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
            except:
                sched = scheduler(optimizer=opt, **scheduler_kwargs)

        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None
        if early_stopping and early_stopping_patience is None:
            logger.debug("Using early stopping with infinite patience")
        elif early_stopping:
            logger.debug("Using early stopping with patience %s", early_stopping_patience)
        else:
            logger.debug("No early stopping")

        n_losses = len(loss_labels)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        n_epochs_verbose = self._set_verbosity(epochs, verbose)

        logger.debug("Beginning main training loop")
        losses_train, losses_val = [], []

        # Resuming training
        if initial_epoch is None:
            initial_epoch = 0
        else:
            logger.info("Resuming with epoch %s", initial_epoch + 1)
            for _ in range(initial_epoch):
                sched.step()  # Hacky, but last_epoch doesn't work when not saving the optimizer state

        # Initial callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(-1, self.model, 0.0, 0.0, last_batch=self.last_batch)

        # Loop over epochs
        for i_epoch in range(initial_epoch, epochs):
            logger.debug("Training epoch %s / %s", i_epoch + 1, epochs)

            #Evaluate the singular values:
            if (i_epoch+1) % 10 == 0:
                first_val_batch = next(iter(val_loader))
                sing_values, error_flag = self.evaluate_singular_values(first_val_batch)
                if error_flag:
                    print("A numerical error in the SVD calculation has occurred during the evaluation!")
                    print('The training is stopped here.')
                    break
                else:
                    np.save(os.path.join(save_dir, 'sing_values_' + '%d' % i_epoch +'.npy'), sing_values)

                    #sub_writer.add_scalar('avg_log_prob', avg_log_prob, epoch)
                    image = plot_spectrum(sing_values, return_tensor=True, ground_truth=latent_dim)
                    writer.add_image('Specturms', image, i_epoch)

                    image = plot_spectrum(sing_values, return_tensor=True, ground_truth=latent_dim, yaxis='log')
                    writer.add_image('Specturms (log-yaxis)', image, i_epoch)
            

            # LR schedule
            if sched is not None:
                logger.debug("Learning rate: %s", sched.get_last_lr())

            try:
                loss_train, loss_val, loss_contributions_train, loss_contributions_val = self.epoch(
                    i_epoch,
                    train_loader,
                    val_loader,
                    opt,
                    loss_functions,
                    loss_weights,
                    clip_gradient,
                    parameters,
                    sig2,
                    noise_type,
                    forward_kwargs=forward_kwargs,
                    custom_kwargs=custom_kwargs,
                    compute_loss_variance=compute_loss_variance,
                    writer=writer
                )
                losses_train.append(loss_train)
                losses_val.append(loss_val)

                #tensorboard logging
                writer.add_scalar('Train_Loss_Epoch', loss_train, i_epoch)
                writer.add_scalar('Val_Loss_Epoch', loss_val, i_epoch)

            except NanException:
                logger.info("Ending training during epoch %s because NaNs appeared", i_epoch + 1)
                raise

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(best_loss, best_model, best_epoch, loss_val, i_epoch, early_stopping_patience)
                except EarlyStoppingException:
                    logger.info("Early stopping: ending training after %s epochs", i_epoch + 1)
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=verbose_epoch)

            # Callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(i_epoch, self.model, loss_train, loss_val, last_batch=self.last_batch)


            if sched is not None:
                #tensorboard logging
                logger.debug("Learning rate: %.5f", sched.get_last_lr()[0])
                writer.add_scalar('Learning_Rate', sched.get_last_lr()[0], i_epoch)
                sched.step()

                # Check learning rate threshold
                current_lr = sched.get_last_lr()[0]
                lr_threshold = 1e-7
                if current_lr < lr_threshold:
                    logger.info("Learning rate dropped below threshold, ending training.")
                    break

                if restart_scheduler is not None and (i_epoch + 1) % restart_scheduler == 0:
                    try:
                        sched = scheduler(optimizer=opt, T_max=epochs_per_scheduler, **scheduler_kwargs)
                    except:
                        sched = scheduler(optimizer=opt, **scheduler_kwargs)

        if early_stopping and len(losses_val) > 0:
            self.wrap_up_early_stopping(best_model, losses_val[-1], best_loss, best_epoch)

        logger.debug("Training finished")
        writer.close() #close the tensorboard writer

        return np.array(losses_train), np.array(losses_val)

    def epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        clip_gradient,
        parameters,
        sig2,
        noise_type,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
        writer=None
    ):
        n_losses = len(loss_weights)
        
        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0
        
         # Create tqdm progress bar for training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", position=1, leave=True)

        for i_batch, batch_data in train_pbar:
            if i_batch == 0 and i_epoch == 0:
                x = batch_data[0]
                print('x mean',x[0,:].mean())
                print('x std',x[0,:].std())
                # print('first batch stats',x[0,0,0,:])
                self.first_batch(batch_data, forward_kwargs)
                #np.save(create_filename("sample", "first_batch", args), batch_data.detach().cpu().numpy())
            
            #print('batch_data size: ', batch_data.size())
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters,sig2,noise_type,i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)
            
            # Update tqdm description with train_loss
            train_pbar.set_description(f"Training (loss: {batch_loss:.4f})")
            writer.add_scalar('Train_Loss_Step', batch_loss, i_epoch * len(train_loader) + i_batch)
            train_pbar.refresh()

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", position=2, leave=True)

            for i_batch, batch_data in val_pbar:
                batch_loss, batch_loss_contributions = self.batch_val(batch_data, loss_functions, loss_weights,None,noise_type,i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)
                # Update tqdm description with val_loss
                val_pbar.set_description(f"Validation (loss: {batch_loss:.4f})")
                writer.add_scalar('Val_Loss_Step', batch_loss, i_epoch * len(val_loader) + i_batch)
                val_pbar.refresh()

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def partial_epoch(
        self,
        i_epoch,
        train_loader,
        val_loader,
        optimizer,
        loss_functions,
        loss_weights,
        parameters,
        sig2 = None,
        noise_type = None,
        clip_gradient=None,
        i_batch_start_train=0,
        i_batch_start_val=0,
        forward_kwargs=None,
        custom_kwargs=None,
        compute_loss_variance=False,
    ):
        if compute_loss_variance:
            raise NotImplementedError

        n_losses = len(loss_weights)
        assert len(loss_functions) == n_losses, "{} loss functions, but {} weights".format(len(loss_functions), n_losses)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = [] if compute_loss_variance else 0.0

        i_batch = i_batch_start_train

        for batch_data in train_loader:
            if i_batch == 0 and i_epoch == 0:
                self.first_batch(batch_data, forward_kwargs)
            batch_loss, batch_loss_contributions = self.batch_train(
                batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters,sig2,noise_type,i_epoch,forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs
            )
            if compute_loss_variance:
                loss_train.append(batch_loss)
            else:
                loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                loss_contributions_train[i] += batch_loss_contribution

            self.report_batch(i_epoch, i_batch, True, batch_data, batch_loss)

        i_batch += 1

        loss_contributions_train /= len(train_loader)
        if compute_loss_variance:
            loss_train = np.array([np.mean(loss_train), np.std(loss_train)])
        else:
            loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = [] if compute_loss_variance else 0.0

            i_batch = i_batch_start_val

            for batch_data in val_loader:
                batch_loss, batch_loss_contributions = self.batch_val(batch_data, loss_functions, loss_weights, None,  noise_type, i_epoch,forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
                if compute_loss_variance:
                    loss_val.append(batch_loss)
                else:
                    loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions[:n_losses]):
                    loss_contributions_val[i] += batch_loss_contribution

                self.report_batch(i_epoch, i_batch, False, batch_data, batch_loss)

            i_batch += 1

            loss_contributions_val /= len(val_loader)
            if compute_loss_variance:
                loss_val = np.array([np.mean(loss_val), np.std(loss_val)])
            else:
                loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def first_batch(self, batch_data):
        pass

    def batch_train(self, batch_data, loss_functions, loss_weights, optimizer, clip_gradient, parameters, sig2, noise_type, i_epoch , forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions, sig2, noise_type, i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)
        self.optimizer_step(optimizer, loss, clip_gradient, parameters)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, batch_data, loss_functions, loss_weights,sig2,noise_type, i_epoch, forward_kwargs=None, custom_kwargs=None):
        loss_contributions = self.forward_pass(batch_data, loss_functions, None, noise_type, i_epoch, forward_kwargs=forward_kwargs, custom_kwargs=custom_kwargs)
        loss = self.sum_losses(loss_contributions, loss_weights)

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def forward_pass(self, batch_data, loss_functions, sig2, noise_type, forward_kwargs=None, custom_kwargs=None):
        """
        Forward pass of the model. Needs to be implemented by any subclass.

        Parameters
        ----------
        batch_data : OrderedDict with str keys and Tensor values
            The data of the minibatch.

        loss_functions : list of function
            Loss functions.

        Returns
        -------
        losses : list of Tensor
            Losses as scalar pyTorch tensors.

        """
        raise NotImplementedError

    def report_batch(self, i_epoch, i_batch, train, batch_data, batch_loss):
        pass


class ForwardTrainer(Trainer):
    """ Trainer for likelihood-based flow training when the model is not conditional. """

    def evaluate_singular_values(self, batch_data, points=1):
        def get_encoder_fn(model):
            def encoder_fn(x):
                _, h_manifold, h_orthogonal, _, _ = self.model._encode(x)
                return torch.cat((h_manifold, h_orthogonal), -1)
            return encoder_fn

        error_flag = False  # initialize the error flag
        try:
            self.model.eval()
            encoder_fn = get_encoder_fn(self.model)
            batchsize = points  # only considering first two points
            loader = torch.utils.data.DataLoader(batch_data, batch_size=1, shuffle=False)
            resolution = torch.prod(torch.tensor(batch_data[0].size()))
            sing_values = np.zeros([batchsize, resolution])

            # Use a loop to iterate over only the first two points
            for i, batch_data in tqdm(enumerate(loader, 0)):
                if i >= batchsize:  # If i is greater than or equal to 2, break the loop.
                    break

                
                start_time = time.time()  # Start the timer

                x = batch_data[0:1, :].float().to(self.device)
                x_ = torch.autograd.Variable(x)
                jac_ = torch.autograd.functional.jacobian(encoder_fn, x_)
                
                end_time = time.time()  # End the timer
                duration = end_time - start_time  # Calculate the time difference
                print(f"Jacobian calculation: took {duration:.4f} seconds")

                start_time = time.time()
                jac_mat = jac_.reshape([resolution, resolution])
                U, S, V = torch.svd(jac_mat)
                sing_values[i, :] = S.detach().cpu().numpy()

                end_time = time.time()  # End the timer
                duration = end_time - start_time  # Calculate the time difference
                print(f"SVD calculation: took {duration:.4f} seconds")

                
                


            self.model.train()

            return sing_values, error_flag
        except Exception as e:
            error_flag = True
            print(f"Error encountered: {e}")
            return None, error_flag

    def first_batch(self, batch_data, forward_kwargs):
        if self.multi_gpu:
            x, y = batch_data
            if len(x.size()) < 2:
                x = x.view(x.size(0), -1)
            x = x.to(self.device, self.dtype)
            self.model(x[: x.shape[0] // torch.cuda.device_count(), ...], **forward_kwargs)

    def add_noise(self,noise_type,x,sig2): #dataset,
        if noise_type == 'gaussian':            
            noise = np.sqrt(sig2) * torch.randn(x.shape,device=self.device,requires_grad = False).to(self.device)

        elif noise_type == 'uniform':   # torch.rand_like(img)
            noise = np.sqrt(sig2) * torch.rand_like(x,device=self.device,requires_grad = False).to(self.device)
        
        # elif noise_type == 'true_normal':            
        #     if dataset == 'thin_spiral':
        #         norm = torch.norm(x,dim=1).reshape([x.shape[0],1])
        #         z = 3 * norm
        #         e_r = x / norm
        #         R = torch.tensor([[0,-1],[1,0]]).float()
        #         e_phi = +1*torch.matmul(e_r,R)
        #         x_norm = (e_r + z * e_phi)/3 
        #         scale = np.sqrt(sig2) * torch.randn([x.shape[0]])
        #         noise_ = scale.reshape([x.shape[0],1]) * x_norm  / torch.norm(x_norm,dim=1).reshape([x.shape[0],1])
        #         noise = torch.matmul(noise_,R)
        #     elif dataset == 'circle':
        #         noise = x
                
        # elif noise_type == 'model_nn':
        #     x_normal = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
        #     norm = torch.norm(x_normal,dim=1).reshape([x.shape[0],1])
        #     x_normal_norm = (x_normal / norm)
        #     scale = np.sqrt(sig2) * torch.randn([x_normal.shape[0]])
        #     noise = scale.reshape([x_normal.shape[0],1]) * x_normal_norm            
         
        # elif noise_type == 'R3_nn':
        #     noise = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
            
        return noise

    def forward_pass(self, batch_data, loss_functions, sig2, noise_type, i_epoch, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}
        
        
        x = batch_data #batch_data[0]

        self._check_for_nans("Training data", x)
        
        if len(x.size()) < 2:
            #logger.info('x size is <2')
            x = x.view(x.size(0), -1)
        x = x.to(self.device, self.dtype)
        #logger.info('First batch coordinate %s',x[0,0,0,0])
        if self.multi_gpu:
            results = nn.parallel.data_parallel(self.model, x, module_kwargs=forward_kwargs)
        else:
            if sig2 is not None:
                noise = self.add_noise(noise_type,x,sig2)  #'thin_spiral'
                x_tilde =  x + noise

            else: x_tilde = x
            results = self.model(x_tilde, **forward_kwargs)
        if len(results) == 4:
            x_reco, log_prob, u, hidden = results
        else:
            x_reco, log_prob, u = results
            hidden = None
        #logger.info('First x_reco %s',x_reco[0,0,0,0])
        
        self._check_for_nans("Reconstructed data", x_reco, fix_until=5)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=5)
        if x.size(0) >= 15:
            self.last_batch = {
                "x": x.detach().cpu().numpy(),
                "x_reco": x_reco.detach().cpu().numpy(),
                "log_prob": None if log_prob is None else log_prob.detach().cpu().numpy(),
                "u": u.detach().cpu().numpy(),
            }

        losses = [loss_fn(x_reco, x, log_prob, hidden=hidden) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class ConditionalForwardTrainer(Trainer):
    """ Trainer for likelihood-based flow training for conditional models. """

    def add_noise(self,dataset,noise_type,x,sig2):
        if noise_type == 'gaussian':            
            noise = np.sqrt(sig2) * torch.randn(x.shape,device=self.device,requires_grad = False).to(self.device)
        
        elif noise_type == 'true_normal':            
            if dataset == 'thin_sprial':
                norm = torch.norm(x,dim=1).reshape([x.shape[0],1])
                z = 3 * norm
                e_r = x / norm
                R = torch.tensor([[0,-1],[1,0]]).float()
                e_phi = +1*torch.matmul(e_r,R)
                x_norm = (e_r + z * e_phi)/3 
                scale = np.sqrt(sig2) * torch.randn([x.shape[0]])
                noise_ = scale.reshape([x.shape[0],1]) * x_norm  / torch.norm(x_norm,dim=1) 
                noise = torch.matmul(noise_,R)
            elif dataset == 'circle':
                noise = x
                
        elif noise_type == 'model_nn':
            x_normal = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
            norm = torch.norm(x_normal,dim=1).reshape([x.shape[0],1])
            x_normal_norm = (x_normal / norm)
            scale = np.sqrt(sig2) * torch.randn([x_normal.shape[0]])
            noise = scale.reshape([x_normal.shape[0],1]) * x_normal_norm            
         
        elif noise_type == 'R3_nn':
            noise = self.model.normal_sampling(x).detach().clone().to(self.device, self.dtype) - x
            
        return noise

    def forward_pass(self, batch_data, loss_functions,sig2,noise_type, i_epoch, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x = batch_data[0]
        params = batch_data[1]

        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        if len(params.size()) < 2:
            params = params.view(params.size(0), -1)

        x = x.to(self.device, self.dtype)
        params = params.to(self.device, self.dtype)
        self._check_for_nans("Training data", x, params)

        if self.multi_gpu:
            forward_kwargs["context"] = params
            results = nn.parallel.data_parallel(self.model, x, module_kwargs=forward_kwargs)
        else:
            if sig2 is not None:
                noise = self.add_noise('thin_spiral',noise_type,x,sig2)
                x_tilde =  x + noise
            else: x_tilde = x
            
            results = self.model(x, context=params, **forward_kwargs)

        if len(results) == 4:
            x_reco, log_prob, u, hidden = results
        else:
            x_reco, log_prob, u = results
            hidden = None

        self._check_for_nans("Reconstructed data", x_reco)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=5)
        if x.size(0) >= 15:
            self.last_batch = {
                "x": x.detach().cpu().numpy(),
                "params": params.detach().cpu().numpy(),
                "x_reco": x_reco.detach().cpu().numpy(),
                "log_prob": None if log_prob is None else log_prob.detach().cpu().numpy(),
                "u": u.detach().cpu().numpy(),
            }

        losses = [loss_fn(x_reco, x, log_prob, hidden=hidden) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class SCANDALForwardTrainer(Trainer):
    """ Trainer for likelihood-based flow training for conditional models with SCANDAL-like loss. """

    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x, params, t_xz = batch_data

        if len(x.size()) < 2:
            x = x.view(x.size(0), -1)
        if len(params.size()) < 2:
            params = params.view(params.size(0), -1)

        x = x.to(self.device, self.dtype)
        params = params.to(self.device, self.dtype)
        t_xz = t_xz.to(self.device, self.dtype)
        self._check_for_nans("Training data", x, params, t_xz)

        if not params.requires_grad:
            params.requires_grad = True

        if self.multi_gpu:
            x_reco, log_prob, _ = nn.parallel.data_parallel(self.model, x, module_kwargs={"context": params})
        else:
            x_reco, log_prob, _ = self.model(x, context=params, **forward_kwargs)

        (t,) = grad(log_prob, params, grad_outputs=torch.ones_like(log_prob.data), only_inputs=True, create_graph=True)

        self._check_for_nans("Reconstructed data", x_reco)
        if log_prob is not None:
            self._check_for_nans("Log likelihood", log_prob, fix_until=5)

        losses = [loss_fn(x_reco, x, log_prob, t, t_xz) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class AdversarialTrainer(Trainer):
    """ Trainer for adversarial (OT) flow training when the model is not conditional. """

    # TODO: multi-GPU support
    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x = batch_data[0]
        batch_size = x.size(0)

        if len(x.size()) < 2:
            x = x.view(batch_size, -1)
        x = x.to(self.device, self.dtype)
        self._check_for_nans("Training data", x)

        x_gen = self.model.sample(n=batch_size, **forward_kwargs)
        self._check_for_nans("Generated data", x_gen)

        losses = [loss_fn(x_gen, x, None) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


class ConditionalAdversarialTrainer(AdversarialTrainer):
    """ Trainer for adversarial (OT) flow training and conditional models. """

    # TODO: multi-GPU support
    def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
        if forward_kwargs is None:
            forward_kwargs = {}

        x = batch_data[0]
        params = batch_data[1]
        batch_size = x.size(0)

        if len(x.size()) < 2:
            x = x.view(batch_size, -1)
        if len(params.size()) < 2:
            params = params.view(batch_size, -1)
        self._check_for_nans("Training data", x, params)

        x = x.to(self.device, self.dtype)
        params = params.to(self.device, self.dtype)

        x_gen = self.model.sample(n=batch_size, context=params, **forward_kwargs)
        self._check_for_nans("Generated data", x_gen)

        losses = [loss_fn(x_gen, x, None) for loss_fn in loss_functions]
        self._check_for_nans("Loss", *losses)

        return losses


# class VarDimForwardTrainer(ForwardTrainer):
#     """ Trainer for likelihood-based flow training for PIE with variable epsilons and non-conditional models. """
#
#     def train(
#         self,
#         dataset,
#         loss_functions,
#         loss_weights=None,
#         loss_labels=None,
#         epochs=50,
#         batch_size=100,
#         optimizer=optim.Adam,
#         optimizer_kwargs=None,
#         initial_lr=1.0e-3,
#         scheduler=optim.lr_scheduler.CosineAnnealingLR,
#         scheduler_kwargs=None,
#         restart_scheduler=None,
#         validation_split=0.25,
#         early_stopping=True,
#         early_stopping_patience=None,
#         clip_gradient=1.0,
#         verbose="some",
#         parameters=None,
#         callbacks=None,
#         forward_kwargs=None,
#         custom_kwargs=None,
#         compute_loss_variance=False,
#         l1=0.0,
#         l2=0.0,
#     ):
#         # Prepare inputs
#         if custom_kwargs is None:
#             custom_kwargs = dict()
#         if l1 is not None:
#             custom_kwargs["l1"] = l1
#         if l2 is not None:
#             custom_kwargs["l2"] = l2
#
#         n_losses = len(loss_functions) + 1
#         if loss_labels is None:
#             loss_labels = [fn.__name__ for fn in loss_functions]
#         loss_labels.append("Regularizer")
#         loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights + [1.0]
#
#         return super().train(
#             dataset,
#             loss_functions,
#             loss_weights,
#             loss_labels,
#             epochs,
#             batch_size,
#             optimizer,
#             optimizer_kwargs,
#             initial_lr,
#             scheduler,
#             scheduler_kwargs,
#             restart_scheduler,
#             validation_split,
#             early_stopping,
#             early_stopping_patience,
#             clip_gradient,
#             verbose,
#             parameters,
#             callbacks,
#             forward_kwargs,
#             custom_kwargs,
#             compute_loss_variance=compute_loss_variance,
#         )
#
#     def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
#         losses = super().forward_pass(batch_data, loss_functions, forward_kwargs)
#
#         if custom_kwargs is not None:
#             l1 = custom_kwargs.get("l1", 0.0)
#             l2 = custom_kwargs.get("l2", 0.0)
#             reg = self.model.latent_regularizer(l1, l2)
#             losses.append(reg)
#
#         return losses
#
#     def report_epoch(self, i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
#         logging_fn = logger.info if verbose else logger.debug
#         super().report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose)
#
#         logging_fn("           latent dim {:>8d}".format(self.model.calculate_latent_dim()))
#         logger.debug("           stds        {}".format(self.model.latent_stds().detach().numpy()))
#
#
# class ConditionalVarDimForwardTrainer(ConditionalForwardTrainer):
#     """ Trainer for likelihood-based flow training for PIE with variable epsilons and conditional models. """
#
#     def train(
#         self,
#         dataset,
#         loss_functions,
#         loss_weights=None,
#         loss_labels=None,
#         epochs=50,
#         batch_size=100,
#         optimizer=optim.Adam,
#         optimizer_kwargs=None,
#         initial_lr=1.0e-3,
#         scheduler=optim.lr_scheduler.CosineAnnealingLR,
#         scheduler_kwargs=None,
#         restart_scheduler=None,
#         validation_split=0.25,
#         early_stopping=True,
#         early_stopping_patience=None,
#         clip_gradient=1.0,
#         verbose="some",
#         parameters=None,
#         callbacks=None,
#         forward_kwargs=None,
#         custom_kwargs=None,
#         compute_loss_variance=False,
#         l1=0.0,
#         l2=0.0,
#     ):
#         # Prepare inputs
#         if custom_kwargs is None:
#             custom_kwargs = dict()
#         if l1 is not None:
#             custom_kwargs["l1"] = l1
#         if l2 is not None:
#             custom_kwargs["l2"] = l2
#
#         n_losses = len(loss_functions) + 1
#         if loss_labels is None:
#             loss_labels = [fn.__name__ for fn in loss_functions]
#         loss_labels.append("Regularizer")
#         loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights + [1.0]
#
#         return super().train(
#             dataset,
#             loss_functions,
#             loss_weights,
#             loss_labels,
#             epochs,
#             batch_size,
#             optimizer,
#             optimizer_kwargs,
#             initial_lr,
#             scheduler,
#             scheduler_kwargs,
#             restart_scheduler,
#             validation_split,
#             early_stopping,
#             early_stopping_patience,
#             clip_gradient,
#             verbose,
#             parameters,
#             callbacks,
#             forward_kwargs,
#             custom_kwargs,
#             compute_loss_variance=compute_loss_variance,
#         )
#
#     def forward_pass(self, batch_data, loss_functions, forward_kwargs=None, custom_kwargs=None):
#         losses = super().forward_pass(batch_data, loss_functions, forward_kwargs)
#
#         if custom_kwargs is not None:
#             l1 = custom_kwargs.get("l1", 0.0)
#             l2 = custom_kwargs.get("l2", 0.0)
#             reg = self.model.latent_regularizer(l1, l2)
#             losses.append(reg)
#
#         return losses
#
#     def report_epoch(self, i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False):
#         logging_fn = logger.info if verbose else logger.debug
#         super().report_epoch(i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose)
#
#         logging_fn("           latent dim {:>8d}".format(self.model.calculate_latent_dim()))
#         logger.debug("           stds        {}".format(self.model.latent_stds().detach().numpy()))