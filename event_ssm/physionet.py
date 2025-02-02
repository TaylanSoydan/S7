import os, sys
os.environ['PYTHONPATH'] = 'data/old_home/tsoydan/RPG'
sys.path.append('/data/old_home/tsoydan/RPG')

from easyneuralode.physionet_data import init_physionet_data


import argparse
import collections
import os
import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


REGS = ["r2", "r3", "r4", "r5"]

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_root', type=str, default="./")
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--lam_w', type=float, default=0)
parser.add_argument('--atol', type=float, default=1.4e-8)
parser.add_argument('--rtol', type=float, default=1.4e-8)
parser.add_argument('--init_step', type=float, default=1.)
parser.add_argument('--reg', type=str, choices=['none'] + REGS, default='none')
parser.add_argument('--test_freq', type=int, default=640)
parser.add_argument('--save_freq', type=int, default=640)
parser.add_argument('--dirname', type=str, default='tmp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_count_nfe', action="store_true")
parse_args = parser.parse_args()

rng = jax.random.PRNGKey(32)
ds_train, ds_test, meta = init_physionet_data(rng,parse_args)

for _ in range(10):
    print(_)
    sample = next(iter(ds_train))

    for k,v in sample.items():
        print(k,v.shape)



