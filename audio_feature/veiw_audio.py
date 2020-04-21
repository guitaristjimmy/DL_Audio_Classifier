import librosa
import wave
from matplotlib import pyplot as plt
import numpy as np
# import IPython.display as ipd


def normalize(data):
    max, min = data.max(), data.min()
    for x in range(0, len(data)):
        data[x] = 2*(data[x]-min)/(max-min) - 1
    return data


if __name__ == '__main__':
    with wave.open(f='./dataset/train/bird/00b01445_nohash_0.wav', mode='rb') as w:
        wave_params = w.getparams()
    print('channels = ', wave_params[0])
    print('bit depth = ', wave_params[1]*8)
    print('samplerates = ', wave_params[2])
    print('length = ', wave_params[3]/wave_params[2])
    print('comp = ', wave_params[5])

    audio_data, sr = librosa.load(path='./dataset/train/bird/00b01445_nohash_0.wav', sr=None)

    print(audio_data.shape)
    print(audio_data)

    plt.figure(figsize=(15, 8))
    plt_1 = plt.subplot(2, 2, 1)
    plt_1.set_title('Original audio')
    plt.plot(audio_data)
    plt.ylim([-1, 1])
    plt.xlabel('Time(samples)')
    plt.ylabel('Amp')
    plt.tight_layout()
    # audio_display = ipd.display(ipd.Audio(data=audio_data, rate=sr)) # in jupyter notebook

    data_normalized = normalize(audio_data)

    plt_2 = plt.subplot(2, 2, 2)
    plt_2.set_title('Normalized audio')
    plt.plot(audio_data)
    plt.xlabel('Time(samples)')
    plt.ylabel('Amp')
    plt.tight_layout()

    dt = 1/sr
    f = np.arange(0, len(data_normalized))*(sr/len(data_normalized))

    fft_origin = np.fft.fft(audio_data)
    fft_freq = np.fft.fftfreq(n=len(audio_data), d=dt)

    plt_3 = plt.subplot(2, 2, 3)
    plt_3.set_title('FFT')
    plt.plot(fft_freq, np.abs(fft_origin))
    plt.xlabel('Freq')
    plt.ylabel('Magnitude')
    plt.xlim([0, 4000])
    plt.tight_layout()

    fft_normalize = np.fft.fft(data_normalized)

    plt_4 = plt.subplot(2, 2, 4)
    plt_4.set_title('FFT')
    plt.plot(fft_freq, np.abs(fft_normalize))
    plt.xlabel('Freq')
    plt.ylabel('Magnitude')
    plt.xlim([0, 4000])
    plt.tight_layout()

    plt.show()
