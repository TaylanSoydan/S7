import os, sys
os.environ['PYTHONPATH'] = 'data/old_home/tsoydan/RPG'
sys.path.append('/data/old_home/tsoydan/RPG')

from LEM.src.eigenWorms.data import EigenWorms

train_dataset, test_dataset, valid_dataset = EigenWorms()

for i in range(len(train_dataset)):
    print(train_dataset[i][0].shape)

print(5)