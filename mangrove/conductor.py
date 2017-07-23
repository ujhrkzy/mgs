# -*- coding: utf-8 -*-
import numpy as np
from app.application_context import context
from models.wave import MusicDataConverter
from util.logging.logging import logger
from keras.callbacks import Callback
from models.nn import NeuralNetwork
import os
import glob
from util.collections import is_empty_collection
from util.string_util import is_empty
from numpy import ndarray
from keras.models import Sequential
from models.music_tensor import MusicTensor
from datetime import datetime
from config.app_config import AppConfig

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class LossHistory(Callback):

    def __init__(self):
        self.losses = []

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)


class Conductor(object):

    _weights_file_prefix = "weights_"
    _weights_file_extension = ".h5"
    _weights_file_pattern = _weights_file_prefix + "{0:07d}" + _weights_file_extension
    _music_file_pattern = "{}_epoch_{}.wav"

    def __init__(self):
        self.__config = context.config

    @property
    def _config(self) -> AppConfig:
        return self.__config

    def generate_music(self):
        x_train = np.load(self._config.seed_x_npy)

        weights_file_path = self._get_latest_weights_file_path()
        if is_empty(weights_file_path):
            raise ValueError("weights file is 0.")
        nn_model = NeuralNetwork(x=x_train).create_lstm_nn(weights_file_path=weights_file_path)

        # TODO 別ファイルからseedを作成する
        seed = self._generate_seed_from_tensor(x_train)
        outputs = self._generate_music_from_seed(nn_model=nn_model, seed=seed)
        x_music_tensor = MusicTensor(tensor=x_train, normalize_enable=False)
        x_music_tensor.mean = np.load(self._config.x_mean_npy)
        x_music_tensor.std = np.load(self._config.x_std_npy)
        for i in range(len(outputs)):
            outputs[i] = x_music_tensor.denormalize(outputs[i])

        wave_file_path = self._get_music_file_path()
        converter = MusicDataConverter()
        converter.save_music(music_file_path=wave_file_path, generated_sequence=outputs)
        logger.info("saved.")

    def _get_music_file_path(self):
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")
        total_epoch = self._get_current_epoch(self._get_latest_weights_file_path())
        wave_file_path = self._music_file_pattern.format(current_datetime, total_epoch)
        wave_file_path = os.path.join(self._config.output_music_file_directory, wave_file_path)
        return wave_file_path

    def _generate_music_from_seed(self, nn_model: Sequential, seed: ndarray) -> list:
        seed_sequence = seed.copy()
        output = []
        output.extend([new_seed.copy() for new_seed in seed[0]])
        for i in range(self._config.output_sequence_length):
            new_seed_sequence = nn_model.predict(seed_sequence, batch_size=self._config.batch_size, verbose=0)
            if i == 0:
                output.extend([new_seed.copy() for new_seed in new_seed_sequence[0]])
                # print("")
            else:
                output.append(new_seed_sequence[0][-1].copy())
            new_sequence = new_seed_sequence[0][-1]
            new_sequence = np.reshape(new_sequence, (1, 1, new_sequence.shape[0]))
            seed_sequence = np.concatenate((seed_sequence, new_sequence), axis=1)
            # remove first data
            # axis=1 の 0番目を削除
            # seedSeq.shape = 1,41,11050 を
            # seedSeq.shape = 1,40,11050 に変更
            seed_sequence = np.delete(seed_sequence, 0, 1)
        return output

    def _generate_seed_from_tensor(self, tensor: ndarray) -> ndarray:
        example_length = tensor.shape[0]
        random_int_value = np.random.randint(example_length, size=1)[0]
        random_seed = np.concatenate(tuple([tensor[random_int_value + i] for i in range(self._config.seed_length)]),
                                     axis=0)
        seed_sequence = np.reshape(random_seed, (1, random_seed.shape[0], random_seed.shape[1]))
        return seed_sequence

    def train(self):
        # 19, 40, 11025(config.fft_enable=True -> 22050)
        # x_train shape is (num_train_examples, sequence_length, block_size).
        x_train = np.load(self._config.x_npy)
        y_train = np.load(self._config.y_npy)

        weights_file_path = self._get_latest_weights_file_path()
        nn_model = NeuralNetwork(x=x_train).create_lstm_nn(weights_file_path=weights_file_path)

        logger.info('Start training.')
        current_epoch = 0
        epoch_size = self._config.epoch_size
        nb_epoch = self._config.nb_epoch
        while current_epoch < epoch_size:
            logger.info('current epoch: ' + str(current_epoch))
            history = LossHistory()
            nn_model.fit(x_train, y_train, batch_size=self._config.batch_size, nb_epoch=nb_epoch, verbose=0,
                         validation_split=0.0, callbacks=[history])
            logger.info(history.losses)
            current_epoch += nb_epoch
        logger.info('Training has completed.')
        total_epoch = current_epoch + self._get_current_epoch(weights_file_path)
        nn_model.save_weights(self._get_new_weights_file_path(total_epoch=total_epoch))

    def _get_current_epoch(self, weights_file_path: str):
        if is_empty(weights_file_path):
            return 0
        tmp = os.path.split(weights_file_path)[-1]
        tmp = tmp.replace(self._weights_file_prefix, "")
        tmp = tmp.replace(self._weights_file_extension, "")
        return int(tmp)

    def _get_latest_weights_file_path(self) -> str:
        files = glob.glob(os.path.join(self._config.weight_file_directory, "*.h5"))
        if is_empty_collection(files):
            return None
        files = [file for file in files if os.path.isfile(file)]
        if is_empty_collection(files):
            return None
        files = sorted(files)
        return files[-1]

    def _get_new_weights_file_path(self, total_epoch) -> str:
        file_name = self._weights_file_pattern.format(total_epoch)
        file_name = os.path.join(self._config.weight_file_directory, file_name)
        return file_name

if __name__ == "__main__":
    _weights_file_prefix = "weights_"
    _weights_file_extension = ".h5"
    _weights_file_pattern = _weights_file_prefix + "{0:07d}" + _weights_file_extension
    f = "weights_0000077.h5"
    tmp = f.replace(_weights_file_prefix, "")
    tmp = tmp.replace(_weights_file_extension, "")
    v = int(tmp)
    print(v)