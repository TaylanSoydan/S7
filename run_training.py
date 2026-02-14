import os
from functools import partial

import hydra
import jax
import jax.random
from flax import jax_utils
from flax.training import checkpoints
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, open_dict

from src.dataloading import Datasets
from src.ssm import init_S7SSM
from src.seq_model import BatchClassificationModel, RetrievalModel
from src.train_utils import training_step, evaluation_step, init_model_state
from src.trainer import TrainerModule


def setup_training(key, cfg: DictConfig):
    num_devices = jax.local_device_count()

    create_dataset_fn = Datasets[cfg.task.name]

    print("[*] Loading dataset...")
    train_loader, val_loader, test_loader, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=num_devices,
        **cfg.training,
        **cfg.model.ssm,
    )

    with open_dict(cfg):
        cfg.optimizer.total_steps = cfg.training.num_epochs * len(train_loader) // cfg.optimizer.accumulation_steps
        cfg.optimizer.warmup_steps = cfg.optimizer.warmup_epochs * len(train_loader) // cfg.optimizer.accumulation_steps
        cfg.optimizer.ssm_lr = cfg.optimizer.ssm_base_lr * cfg.training.per_device_batch_size * num_devices * cfg.optimizer.accumulation_steps

    print("[*] Creating model...")
    ssm_init_fn = init_S7SSM(**cfg.model.ssm_init)
    if cfg.task.name == "retrieval-classification":
        model = RetrievalModel(
            ssm=ssm_init_fn,
            num_classes=data.n_classes,
            num_embeddings=data.num_embeddings,
            **cfg.model.ssm,
        )
    else:
        model = BatchClassificationModel(
            ssm=ssm_init_fn,
            num_classes=data.n_classes,
            num_embeddings=data.num_embeddings,
            **cfg.model.ssm,
        )

    print("[*] Initializing model state...")
    single_bsz = cfg.training.per_device_batch_size
    batch = next(iter(train_loader))
    inputs, targets, timesteps, lengths = batch

    state = init_model_state(key, model, inputs[:single_bsz], timesteps[:single_bsz], lengths[:single_bsz], cfg.optimizer)

    if cfg.training.get('from_checkpoint', None):
        print(f'[*] Resuming model from {cfg.training.from_checkpoint}')
        state = checkpoints.restore_checkpoint(cfg.training.from_checkpoint, state)

    if num_devices >= 2:
        print(f"[*] Running training on {num_devices} GPUs")
        state = jax_utils.replicate(state)
        train_step = jax.pmap(
            partial(training_step, distributed=True, loss_type=cfg.training.loss_type),
            axis_name='data',
        )
        eval_step = jax.pmap(
            partial(evaluation_step, distributed=True, loss_type=cfg.training.loss_type),
            axis_name='data',
        )
    else:
        train_step = jax.jit(partial(training_step, loss_type=cfg.training.loss_type))
        eval_step = jax.jit(partial(evaluation_step, loss_type=cfg.training.loss_type))

    trainer = TrainerModule(
        train_state=state,
        training_step_fn=train_step,
        evaluation_step_fn=eval_step,
        world_size=num_devices,
        config=cfg,
    )

    return trainer, train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(config: DictConfig):
    print(om.to_yaml(config))
    with open(os.path.join(config.logging.log_dir, 'config.yaml'), 'w') as f:
        om.save(config, f)

    key = jax.random.PRNGKey(config.seed)
    init_key, dropout_key = jax.random.split(key)

    if jax.local_device_count() > 1:
        dropout_key = jax.random.split(dropout_key, jax.local_device_count())

    trainer, train_loader, val_loader, test_loader = setup_training(key=init_key, cfg=config)

    print("[*] Running training...")
    trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dropout_key=dropout_key,
    )


if __name__ == '__main__':
    main()
