from multiprocessing import Manager, Process
from pandas import DataFrame
import librosa
import os
import numpy as np


class SetFeature:
    def __init__(self, n_fft):
        self.n_fft = n_fft
        self.audio_files = []
        self.folder_path, self.file_path, self.file_names = [], [], []

    def normalize(self, data):
        max, min = data.max(), data.min()
        if max-min == 0:
            return data
        for x in range(0, len(data)):
            data[x] = 2 * (data[x] - min) / (max - min) - 1
        return data

    def load_paths(self, root_path):
        self.folder_path = os.listdir(root_path)
        for i, fp in enumerate(self.folder_path):
            self.file_path.append([])
            self.file_names.append([])
            self.file_names[i] = os.listdir(root_path+'/'+fp)
            self.file_path[i] = [str(root_path + '/' + fp + '/' + n) for n in self.file_names[i]]

        return self.file_names, self.file_path

    def set_feature(self, name, paths, names):
        audio_features = []
        for p in paths:
            a, sr = librosa.load(path=p, sr=None)
            a = a[:sr+1]
            audio = np.zeros(sr)
            audio[:len(a)] = a
            audio = self.normalize(audio)
            audio_features.append(self.extract_features(audio=audio, sr=sr))
        df = DataFrame(data=audio_features, index=names, columns=['n_audio', 'sCentroid', 'sContrast', 'stft',
                                                                  'mel_spectrogram', 'mfcc'])
        # print(df.head(5))
        df.to_pickle(path=name+'.pkl')

    def extract_features(self, audio, sr):
        n_audio = audio
        sCentroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft)
        sContrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft)
        stft = librosa.amplitude_to_db(np.abs(librosa.stft(y=audio, n_fft=self.n_fft)))
        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=self.n_fft))
        mfcc = librosa.feature.mfcc(sr=sr, S=mel_spectrogram, n_mfcc=16)

        return [audio, sCentroid, sContrast, stft, mel_spectrogram, mfcc]


if __name__ == '__main__':

    st = SetFeature(512)
    fnames, fpaths = st.load_paths(root_path='./dataset/train')

    i = 0
    save_at = 'C:/Users/K/Desktop/I_SW/Python_Note/DeepLearningPlactice/dataset/train_prepared/'
    while i < len(fpaths):
        name = save_at + st.folder_path[i]
        p_01 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
        p_01.start()
        print('p_01 start')
        i += 1

        if i < len(fpaths):
            name = save_at + st.folder_path[i]
            p_02 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
            p_02.start()
            print('p_02 start')
        i += 1

        if i < len(fpaths):
            name = save_at + st.folder_path[i]
            p_03 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
            p_03.start()
            print('p_03 start')
        i += 1

        if i < len(fpaths):
            name = save_at + st.folder_path[i]
            p_04 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
            p_04.start()
            print('p_04 start')
        i += 1

        p_01.join()
        print('p_01 join')
        p_02.join()
        print('p_02 join')
        p_03.join()
        print('p_03 join')
        p_04.join()
        print('p_04 join')
    p_01.close()
    p_02.close()
    p_03.close()
    p_04.close()
