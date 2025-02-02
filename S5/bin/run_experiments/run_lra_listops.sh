#python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
#                    --bidirectional=True --blocks=8 --bsz=50 --d_model=128 --dataset=listops-classification \
#                    --epochs=40 --jax_seed=6554595 --lr_factor=3 --n_layers=8 --opt_config=BfastandCdecay \
#                    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=16 --warmup_end=1 --weight_decay=0.04

python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 --batchnorm=True \
                    --bidirectional=True --blocks=2 --bsz=100 --d_model=40 --dataset=listops-classification \
                    --epochs=100 --jax_seed=6554595 --lr_factor=3 --n_layers=3 --opt_config=BfastandCdecay \
                    --p_dropout=0 --ssm_lr_base=0.001 --ssm_size_base=4 --warmup_end=1 --weight_decay=0 \
                    --input_dependent=True --stablessm_a=True --conj_sym=False \
                   