import os
import torch


def load_checkpoint(ckpt_path):
    try:
        state = torch.load(ckpt_path)
    except FileNotFoundError:
        raise "Checkpoint file not found"
    return state


def load_from_checkpoint(
    state,
    net,
    optimizer=None,
    scheduler=None,
    **kwargs,
):
    net.load_state_dict(state['model_state'], **kwargs)
    if optimizer:
        optimizer.load_state_dict(state['opt_state'])
    if scheduler:
        scheduler.load_state_dict(state['sch_state'])
    return net, optimizer, scheduler


def save_checkpoint(log_dir, epoch, model, optimizer, scheduler, train_meter, config):
    state = {
        'epoch': epoch,
        'train_meter': {k: v.avg for k, v in train_meter.items()},
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'sch_state': scheduler.state_dict(),
        'config': config,
    }
    os.makedirs(log_dir, exist_ok=True)
    torch.save(state, os.path.join(log_dir, f'epoch_{epoch}.ckpt'))
