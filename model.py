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
        init_bias=torch.zeros(2049), # TODO: change default
        init_scale=torch.ones(2049),
        nfft=4096,
        nhop=1024,
        seq_duration=6.0
    ):
        super(AmplitudeEstimator, self).__init__()
        # nb_channels = 2
        # sample_rate = 44100
        # nb_timesteps = int(sample_rate * seq_duration)
        # d_in = nb_channels * nfft//2 + nb_timesteps // (nhop+1) + 2

        # amplitude layers
        self.bias_layer = BiasLayer(init_bias, init_scale)
        self.fc_A1 = nn.Sequential(
            nn.Linear(nfft//2+1, 500),
            nn.ReLU()
        )
        self.fc_A2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU()
        )

        # phase layers
        self.fc_phi1 = nn.Sequential(
            nn.Linear(nfft//2+1, 500),
            nn.ReLU()
        )
        self.fc_phi2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU()
        )

        # final layers
        self.fc_final = nn.Sequential(
            nn.Linear(500, 2049),
            nn.ReLU()
        )

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
        # phi = self.fc_phi1(phase_features)
        # phi = self.fc_phi2(phase_features)

        return self.fc_final(A)







class MSS(nn.Module):
    def __init__(self, n_fft=4096, n_hop=1024):
        super(MSS, self).__init__()

        # input transformation
        self.stft = STFT(n_fft, n_hop)

        self.transform = self.stft

        self.estimator = AmplitudeEstimator()

    def forward(self, x):
        """
        Input: (batch_size, nb_channels, nb_timesteps)
        Output:() # TODO: find appropriate output
        """
        X = self.transform(x).transpose(2,3)

        A, phi = F.complex_norm(X), F.angle(X)

        phase_features = self.compute_features(phi)

        A_hat = self.estimator(A, phi)

        phase = torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)
        
        Y_hat = A_hat.unsqueeze(-1) * phase

        return Y_hat.transpose(-3,-2)

    @staticmethod
    def compute_features(phi):
        return phi
