import hydra
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, open_dict
from functools import partial

import jax.random
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from event_ssm.dataloading import Datasets
from event_ssm.ssm import init_S7SSM
from event_ssm.seq_model import BatchClassificationModel
from event_ssm.dataloading import *


def setup_evaluation(cfg: DictConfig):
    num_devices = jax.local_device_count()
    assert cfg.checkpoint, "No checkpoint directory provided. Use checkpoint=<path> to specify a checkpoint."

    # load task specific data
    create_dataset_fn = Datasets[cfg.task.name]

    # Create dataset...
    print("[*] Loading dataset...")
    train_loader, val_loader, test_loader, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=num_devices,
        **cfg.training
    )

    with open_dict(cfg):
        # optax updates the schedule every iteration and not every epoch
        cfg.optimizer.total_steps = cfg.training.num_epochs * len(train_loader) // cfg.optimizer.accumulation_steps
        cfg.optimizer.warmup_steps = cfg.optimizer.warmup_epochs * len(train_loader) // cfg.optimizer.accumulation_steps

        # scale learning rate by batch size
        cfg.optimizer.ssm_lr = cfg.optimizer.ssm_base_lr * cfg.training.per_device_batch_size * num_devices * cfg.optimizer.accumulation_steps

    # load model
    print("[*] Creating model...")
    ssm_init_fn = init_S7SSM(**cfg.model.ssm_init)
    model = BatchClassificationModel(
        ssm=ssm_init_fn,
        num_classes=data.n_classes,
        num_embeddings=data.num_embeddings,
        **cfg.model.ssm,
    )

    # initialize training state
    state = checkpoints.restore_checkpoint(cfg.checkpoint, target=None)
    params = state['params']
    model_state = state['model_state']

    return model, params, model_state, train_loader, val_loader, test_loader


def evaluation_step(
        apply_fn,
        params,
        model_state,
        batch,
        loss_type: str = 'cross_entropy'
):
    """
    Evaluates the loss of the function passed as an argument on a batch, supporting different loss types.

    :param apply_fn: The model's apply function.
    :param params: Model parameters.
    :param model_state: The state of the model (e.g., batch stats).
    :param batch: The data batch consisting of [inputs, targets, integration_timesteps, lengths].
    :param loss_type: The type of loss function to use ('cross_entropy', 'one_hot_cross_entropy', 'mse').
    :return: Dictionary containing metrics {'loss', 'accuracy', 'perplexity' (if applicable)}, and predictions.
    """
    inputs, targets, integration_timesteps, lengths = batch
    logits = apply_fn(
        {'params': params, **model_state},
        inputs, integration_timesteps, lengths,
        False,  # Evaluation mode
    )

    # Compute the loss based on the specified type
    if loss_type == 'cross_entropy':
        loss = optax.softmax_cross_entropy(logits, targets)
    elif loss_type == 'one_hot_cross_entropy':
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(targets, num_classes=logits.shape[-1]))
    elif loss_type == 'mse':
        loss = optax.l2_loss(logits, targets)
    else:
        raise ValueError(f'Unknown loss_type: {loss_type}')

    loss = loss.mean()

    # Compute predictions and accuracy
    preds = jnp.argmax(logits, axis=-1)
    if loss_type == "one_hot_cross_entropy":
        targets_class = targets  
    else:
        targets_class = jnp.argmax(targets, axis=-1)

    if loss_type in {"cross_entropy", "one_hot_cross_entropy"}:
        accuracy = (preds == targets_class).mean()
        perplexity = jnp.exp(loss)  # Compute perplexity only for cross-entropy
    elif loss_type == "mse":
        accuracy = -loss  # No classification-based accuracy; negative loss serves as proxy
        perplexity = None  # Perplexity is not defined for MSE

    # Construct the output metrics dictionary
    metrics = {'loss': loss, 'accuracy': accuracy}
    if perplexity is not None:
        metrics['perplexity'] = perplexity

    return metrics, preds

@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(config: DictConfig):
    print(om.to_yaml(config))

    model, params, model_state, train_loader, val_loader, test_loader = setup_evaluation(cfg=config)
    step = partial(evaluation_step, model.apply, params, model_state)
    step = jax.jit(step)

    # run training
    print("[*] Running evaluation...")
    metrics = {}
    events_per_sample = []
    time_per_sample = []
    targets = []
    predictions = []
    num_batches = 0

    for i, batch in enumerate(test_loader):
        step_metrics, preds = step(batch)

        predictions.append(preds)
        targets.append(jnp.argmax(batch[1], axis=-1))
        time_per_sample.append(jnp.sum(batch[2], axis=1))
        events_per_sample.append(batch[3])

        if not metrics:
            metrics = step_metrics
        else:
            for key, val in step_metrics.items():
                metrics[key] += val
        num_batches += 1

    metrics = {key: jnp.mean(metrics[key] / num_batches).item() for key in metrics}

    print(f"[*] Test accuracy: {100 * metrics['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
