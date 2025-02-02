python run_train.py --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
                    --blocks=2 --bn_momentum=0.9 --bsz=128 --d_model=30 \
                    --dataset=pathfinder-classification  --epochs=2 --jax_seed=8180844 --lr_factor=3 \
                    --n_layers=5 --p_dropout=0.05 --ssm_lr_base=0.0001 \
                    --input_dependent=True --stablessm_a=False --conj_sym=False --mix=False --opt_config=standard \
                    --ssm_size_base=8 --warmup_end=1 --weight_decay=0.03 --dir_name=/data/storage/tsoydan/data/long-range-arena/lra_release/lra_release/pathfinder32


#python run_train.py --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
#                    --blocks=8 --bn_momentum=0.9 --bsz=64 --d_model=192 \
#                    --dataset=pathfinder-classification  --epochs=200 --jax_seed=8180844 --lr_factor=5 \
#                    --n_layers=6 --opt_config=standard --p_dropout=0.05 --ssm_lr_base=0.0009 \
#                    --input_dependent=True --stablessm_a=False \
#                    --ssm_size_base=256 --warmup_end=1 --weight_decay=0.03 --dir_name=/data/storage/tsoydan/data/long-range-arena/lra_release/lra_release/pathfinder32