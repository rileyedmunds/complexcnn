# -*- coding: utf-8 -*-
"""Dataset wrapper for the ESC dataset.

This wrapper tries to mostly replicate the exact setup in the original paper.

Notable exceptions when compared to the original:
- training batches are provided by a perpetual generator which augments the data by randomly
  time-shifting segments on-the-fly instead of a limited number of pre-generated augmentations
- silent segments are not discarded in training/testing

"""

import os

import librosa
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm
# from datascience import datascience

#path stuff:
import sys
sys.path.append('/Users/rileyedmunds/echonet/echonet')

from echonet.datasets.dataset import Dataset
from echonet.utils.generics import load_audio, to_one_hot


class OriginalESC(Dataset):
    """

    """
    def __init__(self, data_dir, work_dir, train_folds, validation_folds, test_folds, esc10=False):
        super().__init__(data_dir, work_dir)

        self.meta = pd.read_csv(data_dir + 'esc50.csv')

        self.train_folds = train_folds
        self.validation_folds = validation_folds
        self.test_folds = test_folds

        self.class_count = 50

        self.bands = 60
        self.segment_length = 202#101

        self.esc10 = esc10
        if self.esc10:
            self.class_count = 10
            self.meta = self.meta[self.meta['esc10']]
            self.categories = pd.unique(self.meta.sort_values('target')['category'])
            self.meta['target'] = self.to_targets(self.meta['category'])
        else:
            self.categories = pd.unique(self.meta.sort_values('target')['category'])

        self.train_meta = self.meta[self.meta['fold'].isin(self.train_folds)]
        self.validation_data.meta = self.meta[self.meta['fold'].isin(self.validation_folds)]
        self.test_data.meta = self.meta[self.meta['fold'].isin(self.test_folds)]

        self._validation_size = len(self.validation_data.meta)
        self._test_size = len(self.test_data.meta)

        self._generate_spectrograms()
        self._populate(self.validation_data)
        self._populate(self.test_data)
        # self._populate_portion(self.test_data)



        #attempt to populate training data:




    def _generate_spectrograms(self):
        for row in tqdm(self.meta.itertuples(), total=len(self.meta)):
            specfile = self.work_dir + row.filename + '.orig.spec.npy'.format(self.bands)

            if os.path.exists(specfile):
                continue

            audio = load_audio(self.data_dir + 'audio/' + row.filename, 22050)
            audio *= 1.0 / np.max(np.abs(audio))

            # spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=1024,
            #                                       hop_length=512, n_mels=self.bands)
            # spec = librosa.logamplitude(spec)

            #complex
            spec = librosa.core.stft(audio, n_fft=1024, hop_length=512, n_mels=self.bands, dtype=np.complex64)


            save_graph(spec)


            np.save(specfile, spec, allow_pickle=False)

    def save_graph(stft):
        import numpy as np
        import matplotlib.pyplot as plt

        librosa.display.specshow(librosa.logamplitude(np.abs(stft)**2, ref_power=np.max),y_axis='log', x_axis='time')

        plt.title('Power spectrogram')
        plt.colorbar(format='%+4.0f dB')
        plt.tight_layout()

        plt.savefig('tester.png')

    # def _generate_spectrograms(self):
    #     for row in tqdm(self.meta.itertuples(), total=len(self.meta)):
    #         specfile = self.work_dir + row.filename + '.orig.spec.npy'.format(self.bands)

    #         if os.path.exists(specfile):
    #             continue

    #         audio = load_audio(self.data_dir + 'audio/' + row.filename, 22050)
    #         audio *= 1.0 / np.max(np.abs(audio))

    #         # spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=1024,
    #         #                                       hop_length=512, n_mels=self.bands)

    #         spec = librosa.core.stft(audio, n_fft=1024, hop_length=512, n_mels=self.bands)

    #         # spec = librosa.logamplitude(spec)
            
    #         np.save(specfile, spec, allow_pickle=False)


    def _populate(self, data):
        X, y, meta = [], [], []

        for row in data.meta.itertuples():
            segments = self._extract_all_segments(row.filename) #generates deltas and gets data out of npy to pandas dataframe
            X.extend(segments)
            y.extend(np.repeat(row.target, len(segments)))
            values = dict(zip(row._fields[1:], row[1:]))
            columns = row._fields[1:]
            rows = [pd.DataFrame(values, columns=columns, index=[0]) for _ in range(len(segments))]
            meta.extend(rows)

        X = np.stack(X)
        y = to_one_hot(np.array(y), self.class_count)
        meta = pd.concat(meta, ignore_index=True)

        if self.data_mean is None:
            self.data_mean = np.mean(X)
            self.data_std = np.std(X)

        X -= self.data_mean
        X /= self.data_std

        data.X = X
        data.y = y
        data.meta = meta

    # #new data population function:
    # def _populate_portion(self, data):
    #     #note: looks like the data is all test, so we will split it into 70/20 for train and test
    #     X, y, meta = [], [], []
    #     X_train, y_train, meta_train = [], [], []

    #     #test (make it 20 (0 through 20))
    #     for row in data.meta.itertuples()[:len(data.meta.itertuples()) * .2]:
    #         segments = self._extract_all_segments(row.filename)
    #         X.extend(segments)
    #         y.extend(np.repeat(row.target, len(segments)))
    #         values = dict(zip(row._fields[1:], row[1:]))
    #         columns = row._fields[1:]
    #         rows = [pd.DataFrame(values, columns=columns, index=[0]) for _ in range(len(segments))]
    #         meta.extend(rows)
            
    #     #test
    #     X = np.stack(X)
    #     y = to_one_hot(np.array(y), self.class_count)
    #     meta = pd.concat(meta, ignore_index=True)

    #     if self.data_mean is None:
    #         self.data_mean = np.mean(X)
    #         self.data_std = np.std(X)

    #     X -= self.data_mean
    #     X /= self.data_std

    #     #test
    #     data.X = X
    #     data.y = y
    #     data.meta = meta

    #     #-------------------------------------------------------------------------------------------------

    #     #train (make it 70 (20 through 90))
    #     for row in data.meta.itertuples()[(len(data.meta.itertuples()) * .2) : (len(data.meta.itertuples()) * .9)]:
    #         segments = self._extract_all_segments(row.filename)
    #         X_train.extend(segments)
    #         y_train.extend(np.repeat(row.target, len(segments)))
    #         values = dict(zip(row._fields[1:], row[1:]))
    #         columns = row._fields[1:]
    #         rows = [pd.DataFrame(values, columns=columns, index=[0]) for _ in range(len(segments))]
    #         meta_train.extend(rows)

    #     #train
    #     X_train = np.stack(X_train)
    #     y_train = to_one_hot(np.array(y_train), self.class_count)
    #     meta_train = pd.concat(meta_train, ignore_index=True)

    #     if self.data_mean is None:
    #         self.data_mean = np.mean(X_train)
    #         self.data_std = np.std(X_train)

    #     X_train -= self.data_mean
    #     X_train /= self.data_std

    #     #train
    #     data.X_train = X_train
    #     data.y_train = y_train
    #     data.meta_train = meta_train

    def _extract_all_segments(self, filename):
        spec = np.load(self.work_dir + filename + '.orig.spec.npy')

        segments = []
        hop_length = self.segment_length // 5
        offset = 0

        while offset < np.shape(spec)[1] - self.segment_length:
            segment = spec[:, offset:offset + self.segment_length]
            delta = self._generate_delta(segment)
            offset += hop_length
            segments.append(np.stack([segment, delta]))

        return segments


    def _extract_segment(self, filename):
        spec = np.load(self.work_dir + filename + '.orig.spec.npy')
        offset = self.RandomState.randint(0, np.shape(spec)[1] - self.segment_length + 1)
        spec = spec[:, offset:offset + self.segment_length]
        delta = self._generate_delta(spec)
        return np.stack([spec, delta])

    def _generate_delta(self, spec):
        # ported librosa v0.3.1. implementation
        window = np.arange(4, -5, -1)
        padding = [(0, 0), (5, 5)]
        delta = np.pad(spec, padding, mode='edge')
        delta = scipy.signal.lfilter(window, 1, delta, axis=-1)
        idx = [Ellipsis, slice(5, -5, None)]
        return delta[idx]
        





    @property
    def input_shape(self):
        return 2, self.bands, self.segment_length

    @property
    def train_size(self):
        return len(self.train_meta)

    @property
    def validation_size(self):
        return self._validation_size

    @property
    def validation_segments(self):
        return len(self.validation_data.meta)

    @property
    def test_size(self):
        return self._test_size

    @property
    def test_segments(self):
        return len(self.test_data.meta)

    def to_categories(self, targets):
        return self.categories[targets]

    def to_targets(self, categories):
        return [np.argmax(self.categories == name) for name in categories]

    def test(self, model):
        return self._score(model, self.test_data)

    def validate(self, model):
        return self._score(model, self.validation_data)

    def iterbatches(self, batch_size):
        itrain = super()._iterrows(self.train_meta)

        while True:
            X, y = [], []

            for i in range(batch_size):
                row = next(itrain)
                X.append(self._extract_segment(row.filename))
                y.append(row.target)

            X = np.stack(X)
            y = to_one_hot(np.array(y), self.class_count)

            X -= self.data_mean
            X /= self.data_std

            yield X, y



    #beyond here is for training, and not used for data creation-----------

    def _score(self, model, data):
        predictions = pd.DataFrame(model.predict(data.X))
        results = pd.concat([data.meta[['filename', 'target']], predictions], axis=1)
        results = results.groupby('filename').aggregate('mean').reset_index()
        results['predicted'] = np.argmax(results.iloc[:, 2:].values, axis=1)
        return np.sum(results['predicted'] == results['target']) / len(results)
