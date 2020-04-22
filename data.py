import random
import warnings

import musdb

import torch
from torch.utils.data import DataLoader, Dataset

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    return audio

class MUSDBDataset(Dataset):
    def __init__(
        self,
        target,
        root=None,
        seq_duration=6.0,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=torch.float32,
        return_path=False,
        **musdb_kwargs
    ):
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated.
        root : str
            root path of MUSDB
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        dtype : numeric type
            data type of torch output tuple x and y
        musdb_kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.
        """
        self.target = target
        self.seq_duration = seq_duration
        self.samples_per_track = samples_per_track
        self.random_track_mix = random_track_mix
        self.source_augmentations = source_augmentations
        self.dtype = dtype
        self.sample_rate = 44000 # NOTE: looks useless

        try:
            self.mus = musdb.DB(root=root, **musdb_kwargs)
        except RuntimeError:
            warnings.warn(
                "No path provided to MUSDBDataset. "
                "Please give one as an argument to the dataset "
                "or export MUSDB_PATH in your .bashrc file to load the whole dataset."
                "Using preview version of MUSDB (only first 7 seconds of tracks)...",
                RuntimeWarning
            )
            self.mus = musdb.DB(download=True, **musdb_kwargs)

        self.split = musdb_kwargs.get('split', 'test')
        self.return_path = return_path

    def __getitem__(self, idx):
        """
        """
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[idx // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(
                    0, track.duration - self.seq_duration
                )
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(
                    track.sources[source].audio.T,
                    dtype=self.dtype
                )
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index(self.target)
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype=self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype=self.dtype
            )
        if self.return_path:
            return x, y, track.path

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


def data_loaders(args):
    """Loads the specified dataset from commandline arguments

    Returns
    -------
        train_dataset, validation_dataset
    """

    source_augmentations = Compose(
        [globals()['_augment_' + aug] for aug in args.source_augmentations]
    )
    dataloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True
    )

    train_dataset = MUSDBDataset(
        args.instrument,
        root=args.dataset,
        samples_per_track=args.samples_per_track,
        seq_duration=args.seq_duration,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        is_wav=args.is_wav,
        subsets='train',
        split='train'
    )
    valid_dataset = MUSDBDataset(
        args.instrument,
        root=args.dataset,
        samples_per_track=1,
        seq_duration=None,
        is_wav=args.is_wav,
        subsets='train',
        split='valid'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_kwargs
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, valid_loader



if __name__ == '__main__':
    root = 'datasets/musdb18'
    dataset = MUSDBDataset(root=root)

    from tqdm import tqdm

    for d in tqdm(dataset):
        print(d)

    print(len(dataset))
