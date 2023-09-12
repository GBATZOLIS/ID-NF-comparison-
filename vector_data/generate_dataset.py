import torch
import numpy as np
from torch.utils.data import random_split, Dataset, DataLoader
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='basic data directory')
parser.add_argument('--dataset', type=str, help='dataset')
parser.add_argument('--n_samples', type=int, default=45000,  help='dataset')
parser.add_argument('--n_spheres', type=int, default=1)
parser.add_argument('--ambient_dim', type=int, default=100)
parser.add_argument('--manifold_dim', type=int, default=10)
parser.add_argument('--noise_std', type=float, default=0.)
parser.add_argument('--embedding_type', type=str, default='random_isometry')
parser.add_argument('--radii', type=list, default=[])
parser.add_argument('--angle_std', type=float, default=-1)
parser.add_argument('--split_ratio', type=float, default=0.88889)

def get_data_generator(dataset, **kwargs):
    if dataset == 'Ksphere':
        return generate_KsphereDataset
    else:
        raise NotImplementedError

def sample_sphere(n_samples, manifold_dim, std=-1):
    def polar_to_cartesian(angles):
        xs = []
        sin_prod=1
        for i in range(len(angles)):
            x_i = sin_prod * torch.cos(angles[i])
            xs.append(x_i)
            sin_prod *= torch.sin(angles[i])
        xs.append(sin_prod)
        return torch.stack(xs)[None, ...]

    if std == -1:
        new_data = torch.randn((n_samples, manifold_dim+1))
        norms = torch.linalg.norm(new_data, dim=1)
        new_data = new_data / norms[:,None]
        return new_data
    else:
        sampled_angles = std * torch.randn((n_samples,manifold_dim))
        return torch.cat([polar_to_cartesian(angles) for angles in sampled_angles], dim=0)    

def generate_KsphereDataset(n_samples, n_spheres, ambient_dim, 
                        manifold_dim, noise_std, embedding_type,
                        radii, angle_std, **kwargs):

    if radii == []:
        radii = [1] * n_spheres

    if isinstance(manifold_dim, int):
        manifold_dims = [manifold_dim] * n_spheres
    elif isinstance(manifold_dim, list):
        manifold_dims = manifold_dim
                
    data = []
    for i in range(n_spheres):
        print('shpere %d' % (i+1))
        manifold_dim = manifold_dims[i]
        new_data = sample_sphere(n_samples, manifold_dim, angle_std)
        new_data = new_data * radii[i]

        if embedding_type == 'random_isometry':
            # random isometric embedding
            randomness_generator = torch.Generator().manual_seed(0)
            embedding_matrix = torch.randn(size=(ambient_dim, manifold_dim+1), generator=randomness_generator)
            q, r = np.linalg.qr(embedding_matrix)
            q = torch.from_numpy(q)
            new_data = (q @ new_data.T).T
        elif embedding_type == 'first':
            # embedding into first manifold_dim + 1 dimensions
            suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
            new_data = torch.cat([new_data, suffix_zeros], dim=1)
        elif embedding_type == 'separating':
            # embbedding which puts spheres in non-intersecting dimensions
            if n_spheres * (manifold_dim + 1) > ambient_dim:
                raise RuntimeError('Cant fit that many spheres. Enusre that n_spheres * (manifold_dim + 1) <= ambient_dim')
            prefix_zeros = torch.zeros((n_samples, i * (manifold_dim + 1)))
            new_data = torch.cat([prefix_zeros, new_data], dim=1)
            suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
            new_data = torch.cat([new_data, suffix_zeros], dim=1)
        elif embedding_type == 'along_axis':
            # embbedding which puts spheres in non-intersecting dimensions
            if (n_spheres - 1) + (manifold_dim + 1) > ambient_dim:
                raise RuntimeError('Cant fit that many spheres.')
            prefix_zeros = torch.zeros((n_samples, i))
            new_data = torch.cat([prefix_zeros, new_data], dim=1)
            suffix_zeros = torch.zeros([n_samples, ambient_dim - new_data.shape[1]])
            new_data = torch.cat([new_data, suffix_zeros], dim=1)    
        else:
            raise RuntimeError('Unknown embedding type.')
                        
        # add noise
        new_data = new_data + noise_std * torch.randn_like(new_data)
        data.append(new_data)

    
    data = torch.cat(data, dim=0)

    if data.is_cuda:
        numpy_array = data.cpu().numpy()
    else:
        numpy_array = data.numpy()

    return numpy_array

if __name__ == '__main__':
    args = parser.parse_args()
    # Convert the parsed arguments to a dictionary
    args_dict = vars(args)

    data_generator = get_data_generator(**args_dict) #get the data generating function
    data = data_generator(**args_dict) #generate the data
    np.random.shuffle(data) #shuffle the data
    print(data.shape)

    #split data into a train and a val dataset
    split_idx = int(args.split_ratio * args.n_samples)
    train_data, test_data = data[:split_idx], data[split_idx:]

    # Save the datasets
    save_dir = os.path.join(args.data_dir, args.dataset)
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_path = os.path.join(save_dir, 'train.npy')
    test_path = os.path.join(save_dir, 'val.npy')

    np.save(train_path, train_data)
    np.save(test_path, test_data)