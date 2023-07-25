from collections import defaultdict

from heat.lib.logger import LOGGER

NoneType = type(None)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_decims=3):
        self.num_decims = num_decims
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.val:.{self.num_decims}f} ({self.avg:.{self.num_decims}f})'

    def summary(self):
        return f'{self.avg:.{self.num_decims}f}'


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class DictAverage(defaultdict):

    def __init__(self, ):
        super().__init__(AverageMeter)

    def update(self, dict_values, n=1):
        for key, item in dict_values.items():
            self[key].update(item, n)

    @property
    def avg(self, ):
        return {key: item.avg for key, item in self.items()}

    @property
    def sum(self, ):
        return {key: item.sum for key, item in self.items()}

    def __str__(self):
        fmtstr_list = [name + ": " + str(meter) for name, meter in self.items()]
        return fmtstr_list

    def summary(self):
        fmtstr_list = [name + ": " + meter.summary() for name, meter in self.items()]
        return fmtstr_list


class ProgressMeter(object):
    def __init__(
        self,
        num_batches: int,
        meter: DictAverage,
        prefix: str = "",
    ) -> NoneType:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meter = meter
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr(batch)]
        entries += [name+": "+str(meter) for name, meter in self.meter.items()]
        LOGGER.info('  '.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += self.meter.summary()
        LOGGER.info(' '.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))

        def batch_fmtstr(batch):
            return f"[{batch:{num_digits}d}/{num_batches:{num_digits}d}]"
        return batch_fmtstr
