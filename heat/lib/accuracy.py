from typing import Tuple, List

import torch

from heat.lib.meters import DictAverage, ProgressMeter


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int] = (1,),
) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:, :k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, val_loader):
    print("############# Validation stage #############")

    model.eval()
    with torch.no_grad():
        meter = DictAverage()
        progress = ProgressMeter(len(val_loader), meter)
        for i, (images, target) in enumerate(val_loader):

            images = images.to(model.fc.weight.device, non_blocking=True)
            target = target.to(model.fc.weight.device, non_blocking=True)

            # compute loss
            output = model(images)

            acc1 = accuracy(output, target, topk=(1,))
            meter["Prec@1"].update(acc1[0].item(), images.size(0))

            if i % 10 == 0:
                progress.display(i)

        progress.display_summary()
