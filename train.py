import argparse
from datetime import datetime
from numpy import load
import os
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import warnings
warnings.formatwarning = lambda msg, cat, fname, lineno, file=None, line=None: f'{fname}:{lineno} {cat.__name__}:{msg}'

import torch
import torch.nn as nn
import torchaudio

from data import data_loaders
from model import MSS
from utils.summary import Writer

def debug():
    # cuda blocking
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # reproducibility
    import random
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # traceback of inf and nan in gradients
    torch.autograd.set_detect_anomaly(True)


@torch.no_grad()
def val(model, data_loader, criterion, device, desc=None):
    model.eval()
    sum_loss = 0
    for x, y in tqdm(data_loader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)

        Y_hat = model(x)
        Y = model.transform(y)
        loss = criterion(Y_hat, Y)

        sum_loss += loss.cpu().item()
        # break

    return Y_hat, sum_loss / len(data_loader)


def train(model, data_loader, criterion, optimizer, device, desc=None):
    model.train()
    sum_loss = 0
    for x, y in tqdm(data_loader, desc=desc, leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        Y_hat = model(x)
        Y = model.transform(y)
        loss = criterion(Y_hat, Y)

        loss.backward()
        optimizer.step()

        sum_loss += loss.cpu().item()

    return sum_loss / len(data_loader)



def main(args):
    # create paths to useful directories
    exp_dir = args.runs_dir / args.name
    checkpoints_dir = exp_dir / 'checkpoints'
    logs_dir = exp_dir / 'logs'
    results_dir = exp_dir / 'results'
    for dir in [checkpoints_dir, logs_dir, results_dir]:
        dir.mkdir(exist_ok=True, parents=True)

    print('Loading dataset...', end='\t', flush=True)
    train_loader, val_loader = data_loaders(args)
    print('Done.')


    print('Initializing model...', end='\t', flush=True)
    # choose how to initialize bias layer
    length = args.nfft//2+1
    if args.init_bias == 'constant':
        init_bias = (torch.zeros(length), torch.ones(length))
    elif args.init_bias == 'mean':
        stats_path = args.dataset / 'stats'
        bias = load(stats_path / f'mean_fft{args.nfft:d}-hop{args.nhop:d}.npy')
        scale = load(stats_path / f'std_fft{args.nfft:d}-hop{args.nhop:d}.npy')
        init_bias = (torch.from_numpy(bias), torch.from_numpy(scale))
    elif args.init_bias == 'random':
        init_bias = (torch.randn(length), torch.exp(torch.randn(length)))
    else:
        raise NotImplementedError

    model = MSS(
        init_bias,
        n_fft=args.nfft,
        n_hop=args.nhop,
        context_frames=args.context_frames,
        window=args.window
    ).to(args.device)
    print('Done.')

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # choose criterion
    criterion = nn.MSELoss()
    ISTFT = model.stft.istft

    # eventually load a previous model
    first_epoch = 1
    if args.load_model:
        path = max(list(checkpoints_dir.glob('*.pth')))
        print(f'Loading {path}...')
        checkpoint = torch.load(path)
        first_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # main loop
    with Writer(args.epochs, logs_dir) as writer:
        for epoch in range(first_epoch, args.epochs+1):
            writer.epoch = epoch

            t0 = time.time()

            train_loss = train(
                model,
                train_loader,
                criterion,
                optimizer,
                args.device,
                desc=writer.epoch_desc + 'Training'
            )
            example, val_loss = val(
                model,
                val_loader,
                criterion,
                args.device,
                desc=writer.epoch_desc + 'Validation'
            )

            # TODO: scheduler
            t1 = time.time()

            loss_dict = {
                'Training loss':    train_loss,
                'Validation loss':  val_loss
            }
            # print results
            writer.display_results(loss_dict, t1-t0)

            # add results to tensorboard
            writer.add_scalars('Reconstruction loss', loss_dict)

            if epoch % args.save_freq == 0:
                t0 = time.time()
                print(f'Saving model in {exp_dir}...', end='\t', flush=True)

                # compute reconstructed audio
                audio_ex = ISTFT(example).squeeze(0)

                # save it in a file
                torchaudio.save(str(results_dir / f'{args.instrument}_{epoch:03d}.wav'), audio_ex, 44100)

                # save it to tensorboard
                writer.add_audio(args.instrument.capitalize(), audio_ex, start=5, duration=30, subsampling=2)

                # save checkpoint
                torch.save(
                    dict(
                        epoch=epoch,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict()
                    ),
                    checkpoints_dir / f'checkpoint_{epoch:03d}.pth'
                )

                t1 = time.time()
                print(f'Done ({t1-t0:.0f} s).')

if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='Music Source Separation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--dataset', type=Path, help='location of the dataset',
                        default='datasets/musdb18')
    base.add_argument('-d', '--debug_mode', help='activate debug mode',
                        action='store_true')
    base.add_argument('-l', '--load_model', type=str, help='name of the model to load',
                        default='')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=50)
    expr.add_argument('--track_ext', type=str, help='choose between AAC or WAV',
                        default='wav', choices=['aac', 'wav'])
    expr.add_argument('--instrument', type=str, help='instrument to be extracted',
                        default='vocals', choices=['vocals', 'bass', 'drums', 'other'])

    param.add_argument('--batch_size', type=int, help='batch size',
                        default=16)
    param.add_argument('--init_bias', type=str, help='initial value of bias layer',
                        default='mean', choices=['constant', 'mean', 'random'])
    param.add_argument('--lr', type=float, help='learning rate of optimizer',
                        default=1e-3)
    param.add_argument('--nfft', type=int, help='STFT fft size and window size',
                        default=4096)
    param.add_argument('--nhop', type=int, help='STFT hop size',
                        default=1024)
    param.add_argument('--samples_per_track', type=int, help='number of samples per track',
                        default=16)
    param.add_argument('--seq_duration', type=float, help='sequence duration in seconds (-1 for full-length)',
                        default=6.0)
    param.add_argument('--source_augmentations', type=str, nargs='+', help='source_augmentations',
                        default=['gain', 'channelswap'])
    param.add_argument('--context_frames', type=int, help='number of frames before and after used in reconstruction',
                        default=5)
    param.add_argument('--window', type=str, help='window used in STFT',
                        default='gaussian', choices=['bartlett', 'gaussian', 'hamming', 'hann'])

    misc.add_argument('--configs_dir', type=Path, help='where config files are stored',
                        default='configs')
    misc.add_argument('--device', type=int, help='which GPU to use (-1 for CPU)',
                        default=0)
    misc.add_argument('--name', type=str, help='name of the experiment (timestamp if not provided)',
                        default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                        default=0)
    misc.add_argument('--runs_dir', type=Path, help='path to logs, checkpoints, results, etc.',
                        default='runs')
    misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
                        default=5)

    args = parser.parse_args()

    args.configs_dir.mkdir(exist_ok=True, parents=True)

    if args.debug_mode:
        debug()

    if args.load_model:
        load_model = args.load_model
        with open(args.configs_dir / (args.load_model+'.pkl'), 'rb') as f:
            args = pickle.load(f) # TODO: do this properly
            args.load_model = load_model

    else:
        if args.name is None:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

        with open(args.configs_dir / (args.name+'.pkl'), 'wb') as f:
            pickle.dump(args, f)


    args.is_wav = args.track_ext == 'wav'

    if args.device > 0:
        if torch.cuda.is_available():
            args.device = torch.device(f'cuda:{args.device:d}')
        else:
            warnings.warn(
                "CUDA is not available on your machine. "
                "Running the algorithm on CPU."
            )
            args.device = torch.device('cpu')
    else:
        args.device = torch.device('cpu')



    t0 = time.time()
    main(args)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
    exit(0)
