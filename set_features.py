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
        for p, n in zip(paths, names):
            print('load :: ', n)
            a, sr = librosa.load(path=p, sr=None)
            a = a[:sr+1]
            audio = np.zeros(sr)
            audio[:len(a)] = a
            audio = self.normalize(audio)
            audio_features = self.extract_features(audio=audio, sr=sr)
            df = DataFrame(data=audio_features, index=['n_audio', 'sCentroid', 'sContrast', 'stft',
                                                       'mel_spectrogram', 'mfcc'], columns=[n]).T
            if os.path.isfile(name+'.csv'):
                df.to_csv(path_or_buf=name+'.csv', index=True, header=False, mode='a', encoding='utf-8')
            else:
                df.to_csv(path_or_buf=name+'.csv', index=True, header=True, mode='w', encoding='utf-8')

    def extract_features(self, audio, sr):
        n_audio = self.data_reshape(audio)

        sCentroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft)
        sCentroid = self.data_reshape(sCentroid)

        sContrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft)
        sContrast = self.data_reshape(sContrast)

        stft = librosa.amplitude_to_db(np.abs(librosa.stft(y=audio, n_fft=self.n_fft)))
        stft = self.data_reshape(stft)

        mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=self.n_fft))
        mel_spectrogram = self.data_reshape(mel_spectrogram)

        mfcc = librosa.feature.mfcc(sr=sr, S=mel_spectrogram, n_mfcc=16)
        mfcc = self.data_reshape(mfcc)

        return [n_audio, sCentroid, sContrast, stft, mel_spectrogram, mfcc]

    def data_reshape(self, data, max_length=32382):
        d = []
        for i in data:
            x = i.flatten()
            d.append(np.concatenate((x, np.zeros(max_length-len(x))), axis=None))
        return np.array(d)


if __name__ == '__main__':

    st = SetFeature(512)
    fnames, fpaths = st.load_paths(root_path='./dataset/valid')

    f = 0
    save_at = 'C:/Users/K/Desktop/I_SW/Python_Note/DeepLearningPlactice/dataset/valid_prepared/'
    while f < len(fpaths):
        process_list = [0]*32
        f_len = len(fpaths[f])
        name = save_at + st.folder_path[f]
        for i in range(0, 8):
            ps = i*4
            pe = ps+4
            for j in range(ps, pe):
                s, e = int(f_len/32)*j, int(f_len/32)*(j+1)
                print('process ', j, ' start', 'range ::', s, e)
                process_list[j] = Process(target=st.set_feature, args=(name, fpaths[f][s:e], fnames[f][s:e]))
                process_list[j].start()
            for j in range(ps, pe):
                process_list[j].join()
                process_list[j].close()
                print('process ', j, ' end')
        f += 1
        # if i < len(fpaths):
        #     name = save_at + st.folder_path[i]
        #     p_02 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
        #     p_02.start()
        #     print('p_02 start')
        # i += 1

        # if i < len(fpaths):
        #     name = save_at + st.folder_path[i]
        #     p_03 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
        #     p_03.start()
        #     print('p_03 start')
        # i += 1
        #
        # if i < len(fpaths):
        #     name = save_at + st.folder_path[i]
        #     p_04 = Process(target=st.set_feature, args=(name, fpaths[i], fnames[i]))
        #     p_04.start()
        #     print('p_04 start')
        # i += 1
        # p_01.join()
        # p_01.close()
        # print('p_01 join')
        # p_02.join()
        # print('p_02 join')
        # p_03.join()
        # print('p_03 join')
        # p_04.join()
        # print('p_04 join')
    # p_02.close()
    # p_03.close()
    # p_04.close()
