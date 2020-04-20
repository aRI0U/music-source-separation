from math import pi

import torch
import torch.nn as nn
import torchaudio.functional as F

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class BiasLayer(nn.Module):
    def __init__(self, init_bias, init_scale):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(init_bias)
        self.scale = nn.Parameter(init_scale)

    def forward(self, x, norm, rescale=False):
        biased_x = x - self.bias if norm else x + self.bias
        if rescale:
            return biased_x / self.scale
        return biased_x


class AmplitudeEstimator(nn.Module):
    def __init__(
        self,
        phase_features_dim,
        init_bias,
        n_fft=4096,
        n_hop=1024,
        seq_duration=6.0
    ):
        super(AmplitudeEstimator, self).__init__()
        # nb_channels = 2
        # sample_rate = 44100
        # nb_timesteps = int(sample_rate * seq_duration)
        # d_in = nb_channels * nfft//2 + nb_timesteps // (nhop+1) + 2

        # amplitude layers
        self.bias_layer = BiasLayer(*init_bias)
        self.fc_A1 = nn.Sequential(
            nn.Linear(n_fft//2+1, 500),
            nn.ReLU()
        )
        self.fc_A2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU()
        )

        # phase layers
        self.fc_phi1 = nn.Sequential(
            nn.Linear(n_fft//2+1, 500),
            nn.ReLU()
        )
        self.fc_phi2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU()
        )

        # combining layers
        self.fc_final = nn.Sequential(
            nn.Linear(500, 2049),
            nn.ReLU()
        )

        self.reshape = nn.Linear(phase_features_dim+1, 1)


    def forward(self, amplitude, phase_features):
        """Estimate the amplitude of the unmixed signal

        Parameters
        ----------
        amplitude: torch.tensor, shape (batch_size, nb_channels, L//(nhop+1)+1, nfft//2+1)
            amplitude of the mixture signal
        phase_features: torch.tensor shape (batch_size, nb_channels, L//(nhop+1)+1, , nfft//2+1)
            phase of the mixture signal

        Returns
        -------
        torch.tensor
            amplitude of the unmixed signal
        """
        # extract features from amplitude
        A = self.bias_layer(amplitude, True, rescale=True)
        A = self.fc_A1(A)
        A = self.fc_A2(A)

        # extract features from phase features
        phi = self.fc_phi1(phase_features)
        phi = self.fc_phi2(phi)
        # print(A.shape, phi.shape)
        features = torch.cat((A.unsqueeze(-2), phi), dim=-2)
        features = self.fc_final(features)
        # print(features.shape)
        features = self.reshape(features.transpose(-2,-1)).squeeze(-1)
        # print(features.shape)
        features = self.bias_layer(features, False)

        return features




class MSS(nn.Module):
    def __init__(
        self,
        init_bias,
        n_fft=4096,
        n_hop=1024
    ):
        super(MSS, self).__init__()

        # self.n_fft = n_fft
        # self.n_hop = n_hop

        # input transformation
        self.stft = STFT(n_fft, n_hop)

        self.transform = self.stft

        # phase preprocessing
        phase_features_dim = 2
        self.phase_shift = nn.Parameter(
            2 * pi * n_hop / n_fft * torch.arange(n_fft//2+1)
        )

        self.estimator = AmplitudeEstimator(
            phase_features_dim,
            init_bias,
            n_fft=n_fft,
            n_hop=n_hop
        )

    def forward(self, x):
        """
        Input: (batch_size, nb_channels, nb_timesteps)
        Output:() # TODO: find appropriate output
        """
        X = self.transform(x).transpose(-3,-2)

        A, phi = F.complex_norm(X), F.angle(X)

        phase_features = self.compute_features(phi)

        A_hat = self.estimator(A, phase_features)

        phase = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)

        Y_hat = A_hat.unsqueeze(-1) * phase

        return Y_hat.transpose(-3,-2)


    def compute_features(self, phi):
        """
        Input: (B, C, T, N)
        Output:(B, C, phase_features_dim, T, N)
        """
        dt_phi = self.derivative(phi, -2)
        df_phi = self.derivative(phi, -1)

        dt_phi *= self.phase_shift
        df_phi -= pi

        d_phi = torch.stack((df_phi, dt_phi), dim=-2)
        d_phi = (d_phi + pi) % (2*pi) - pi

        return d_phi

    @staticmethod
    def derivative(x, dim):
        """Approximate the derivative of a tensor using finite differences

        Parameters
        ----------
        x: torch.tensor, shape (*)
            tensor to be derived
        dim: int
            dimension along the tensor should be derived

        Returns
        -------
        torch.tensor, shape (*)
            derivative of `x`
        """
        dx = x.narrow(dim, 1, x.size(dim)-1) - x.narrow(dim, 0, x.size(dim)-1)
        return torch.cat((dx, dx.narrow(dim, -1, 1)), dim=dim)
