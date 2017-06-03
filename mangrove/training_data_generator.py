# -*- coding: utf-8 -*-
import numpy as np
from app.application_context import context
from models.wave import MusicDataConverter
from util.logging.logging import logger
from models.music_tensor import MusicTensor
import itertools

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class TrainingDataGenerator(object):

    def __init__(self):
        self._config = context.config
        self._converter = MusicDataConverter()

    def generate(self):
        # self._create_wave_files()

        max_sequence_length = self._config.max_sequence_length
        chunks_x, chunks_y = self._load_training_data(max_sequence_length=max_sequence_length)
        music_tensor_x, music_tensor_y = self._create_music_tensor(max_sequence_length=max_sequence_length,
                                                                   chunks_x=chunks_x,
                                                                   chunks_y=chunks_y)
        self._write(music_tensor_x, music_tensor_y)

    def _create_music_tensor(self, max_sequence_length, chunks_x: list, chunks_y: list):
        num_examples = len(chunks_x)
        if self._config.fft_enable:
            # TODO リファクタ
            num_dims_out = self._config.block_size * 2
        else:
            num_dims_out = self._config.block_size

        out_shape = (num_examples, max_sequence_length, num_dims_out)
        x_data = np.zeros(out_shape)
        y_data = np.zeros(out_shape)
        for example_index, sequence_index in itertools.product(range(num_examples), range(max_sequence_length)):
            x_data[example_index][sequence_index] = chunks_x[example_index][sequence_index]
            y_data[example_index][sequence_index] = chunks_y[example_index][sequence_index]
            # TODO ステレオ対応。 [][][2列]になっている。
            # [:, 0]は[row:column]の内、すべての行の0列目を取得
            # columnだけ抽出
            # x_data[example_index][sequence_index] = chunks_x[example_index][sequence_index][:, 0]
            # y_data[example_index][sequence_index] = chunks_y[example_index][sequence_index][:, 0]
        music_tensor_x = MusicTensor(tensor=x_data, normalize_enable=True)
        music_tensor_y = MusicTensor(tensor=y_data, normalize_enable=False)
        music_tensor_y.normalize(music_tensor_x)
        return music_tensor_x, music_tensor_y

    def _write(self, music_tensor_x: MusicTensor, music_tensor_y: MusicTensor):
        logger.info('Flushing to disk...')
        c = self._config
        np.save(c.x_npy, music_tensor_x.tensor)
        np.save(c.y_npy, music_tensor_y.tensor)
        np.save(c.x_mean_npy, music_tensor_x.mean)
        np.save(c.x_std_npy, music_tensor_x.std)
        np.savetxt(c.x_mean_csv, music_tensor_x.mean, delimiter=",")
        np.savetxt(c.x_std_csv, music_tensor_x.std, delimiter=",")
        logger.info("Done.")

    def _create_wave_files(self):
        c = self._config
        for org_mp3, mono_mp3, wave in zip(c.org_mp3_files, c.mono_mp3_files, c.wave_files):
            self._converter.convert_mp3_to_wav(org_mp3_file_path=org_mp3, mono_mp3_file_path=mono_mp3,
                                               wave_file_path=wave)

    def _load_training_data(self, max_sequence_length):
        chunks_x = []
        chunks_y = []
        for file_path in self._config.wave_files:
            logger.info("load file: " + file_path)
            training_data = self._converter.load_training_data(file_path)
            cur_seq = 0
            total_seq = len(training_data.x)
            logger.info("total seq len: " + str(total_seq))
            while cur_seq + max_sequence_length < total_seq:
                chunks_x.append(training_data.x[cur_seq:cur_seq + max_sequence_length])
                chunks_y.append(training_data.y[cur_seq:cur_seq + max_sequence_length])
                cur_seq += max_sequence_length
        return chunks_x, chunks_y

