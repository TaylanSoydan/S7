#python run_train.py --C_init=complex_normal --batchnorm=True --bidirectional=True \
#                    --blocks=16 --bn_momentum=0.9 --bsz=32 --d_model=128 --dataset=pathx-classification \
#                    --dt_min=0.0001 --epochs=75 --jax_seed=6429262 --lr_factor=3 --n_layers=6 \
#                    --opt_config=BandCdecay --p_dropout=0.0 --ssm_lr_base=0.0006 --ssm_size_base=256 \
#                    --warmup_end=1 --weight_decay=0.06

python run_train.py --C_init=complex_normal --batchnorm=True --bidirectional=True \
                    --blocks=4 --bn_momentum=0.9 --bsz=32 --d_model=32 --dataset=pathx-classification \
                    --dt_min=0.0001 --epochs=200 --jax_seed=6429262 --lr_factor=4 --n_layers=3 \
                    --opt_config=BandCdecay --p_dropout=0.0 --ssm_lr_base=0.0006 --ssm_size_base=8 \
                    --warmup_end=1 --weight_decay=0 --input_dependent=True --conj_sym=False --stablessm_a=True