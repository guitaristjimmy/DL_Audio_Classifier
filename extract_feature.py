import librosa
from librosa import display as dp
import matplotlib.pyplot as plt
from veiw_audio import normalize


class ExtractFeature:
    def __init__(self, audio, sample_rate, n_fft):
        self.sr = sample_rate
        self.sCentroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, n_fft=n_fft)
        self.sContrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr, n_fft=n_fft)
        self.stft = librosa.amplitude_to_db(librosa.stft(y=audio, n_fft=n_fft))
        self.mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=n_fft))
        self.mfcc = librosa.feature.mfcc(sr=self.sr, S=self.mel_spectrogram, n_mfcc=16)

    def display_feature(self):
        plt.figure(figsize=(6, 8))
        plt.subplot(5, 1, 1)
        dp.specshow(self.sCentroid, sr=self.sr)
        plt.colorbar(); plt.title('Spectral Centroid'); plt.tight_layout()

        plt.subplot(5, 1, 2)
        dp.specshow(self.sContrast, sr=self.sr)
        plt.colorbar(); plt.title('Spectral Contrast'); plt.tight_layout()

        plt.subplot(5, 1, 3)
        dp.specshow(self.stft, sr=self.sr)
        plt.colorbar(); plt.title('Spectrogram'); plt.tight_layout()

        plt.subplot(5, 1, 4)
        dp.specshow(self.mel_spectrogram, sr=self.sr)
        plt.colorbar(); plt.title('Mel_Spectrogram'); plt.tight_layout()

        plt.subplot(5, 1, 5)
        dp.specshow(self.mfcc, sr=self.sr)
        plt.colorbar(); plt.title('MFCC'); plt.tight_layout()

        plt.show()

    def features(self):
        return [self.sCentroid, self.sContrast, self.stft, self.mel_spectrogram, self.mfcc]


if __name__ == '__main__':
    audio, sr = librosa.load(path='./dataset/train/bed/00f0204f_nohash_0.wav', sr=None)
    ef = ExtractFeature(audio=audio, sample_rate=sr, n_fft=256)
    ef.display_feature()

    normalized_audio = normalize(audio)
    ef_norm = ExtractFeature(audio=normalized_audio, sample_rate=sr, n_fft=256)
    ef_norm.display_feature()
