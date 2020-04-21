from multiprocessing import Process
from multiprocessing import cpu_count
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
        df = DataFrame(columns=['mfcc'])
        for p, n in zip(paths, names):
            # print('load :: ', n)
            a, sr = librosa.load(path=p, sr=None)
            a = a[:sr+1]
            audio = np.zeros(sr)
            audio[:len(a)] = a
            audio = self.normalize(audio)
            audio_features = self.extract_features(audio=audio, sr=sr)
            dfdict = DataFrame({n: audio_features}).T
            dfdict.columns = ['n_audio', 'sCentroid', 'sContrast', 'stft', 'mel_spectrogram', 'mfcc']
            df = df.append(dfdict)
        df['c'] = name.split('/')[-1]
        df.to_pickle(path=name+'.pkl')
        return df

    def extract_features(self, audio, sr):
        n_audio = audio[:]
        sCentroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft)
        sContrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft)
        stft = librosa.amplitude_to_db(np.abs(librosa.stft(y=audio, n_fft=self.n_fft)))
        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=self.n_fft))
        mfcc = librosa.feature.mfcc(sr=sr, S=mel_spectrogram, n_mfcc=16)
        return [n_audio, sCentroid, sContrast, stft, mel_spectrogram, mfcc]

    def data_reshape(self, data, max_length=32382):
        data = data.flatten()
        data = np.concatenate((data,  np.zeros(max_length-len(data))), axis=None)
        return data


if __name__ == '__main__':

    st = SetFeature(512)
    fnames, fpaths = st.load_paths(root_path='./dataset/valid')

    save_at = 'C:/Users/K/Desktop/I_SW/Python_Note/DeepLearningPlactice/dataset/valid_ds/'
    df = st.set_feature(name=save_at+st.folder_path[0], paths=fpaths[0], names=fnames[0])

    f = 0
    num_cpu = cpu_count()
    while f < len(fpaths):
        process_list = []
        for _ in range(0, num_cpu):
            name = save_at + st.folder_path[f]
            process_list.append(Process(target=st.set_feature, args=(name, fpaths[f], fnames[f])))
            process_list[-1].start()
            print(len(process_list), 'process start')
            f += 1
            if f >= len(fpaths):
                break
        for p in process_list:
            p.join()
            p.close()
        print('processes end')
