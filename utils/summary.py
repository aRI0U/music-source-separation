import os

from torch.utils.tensorboard import SummaryWriter
from torchaudio.functional import istft

class Writer(SummaryWriter):
    def __init__(self, last_epoch, *args, **kwargs):
        super(Writer, self).__init__(*args, **kwargs)
        self._epoch = 1
        self.last_epoch = last_epoch

        self.indent = 18

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, i):
        self._epoch = i

    @property
    def epoch_desc(self):
        return f'[Epoch {self.epoch:d}/{self.last_epoch:d}]'.ljust(self.indent)

    def display_results(self, loss_dict, d):
        summary = self.epoch_desc
        for k, v in loss_dict.items():
            summary += f'{k}: {v:.7f}   '
        time_indicator = 'Time elapsed: '+'{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60))
        n_cols = os.get_terminal_size()[0]
        space_left = n_cols - len(summary) % n_cols
        if len(time_indicator) < space_left:
            print(summary + time_indicator.rjust(space_left))
        else:
            print(summary)
            print(time_indicator.rjust(n_cols))


    # tensorboard utils
    def add_scalars(self, *args, **kwargs):
        super(Writer, self).add_scalars(*args, **kwargs, global_step=self.epoch)

    def add_audio(self, tag, sound_tensor, duration=None, subsampling=1):
        if duration:
            sound_tensor = sound_tensor[...,:44100*duration]

        if sound_tensor.ndim == 2:
            # stereo to mono
            sound_tensor = sound_tensor.sum(dim=0)

        if subsampling > 1:
            sound_tensor = sound_tensor[::subsampling]

        super(Writer, self).add_audio(tag, sound_tensor, global_step=self.epoch, sample_rate=44100//subsampling)
