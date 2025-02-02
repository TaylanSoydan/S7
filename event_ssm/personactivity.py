import os, sys
os.environ['PYTHONPATH'] = 'data/old_home/tsoydan/RPG'
sys.path.append('/data/old_home/tsoydan/RPG')

from odelstms.irregular_sampled_datasets import PersonData

PersonData()