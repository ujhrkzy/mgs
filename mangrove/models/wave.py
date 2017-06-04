# -*- coding: utf-8 -*-
from app.application_context import context
from numpy import ndarray
import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from util.logging.logging import logger
import subprocess

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class TrainingData(object):

    def __init__(self, x: list, y: list, bit_rate):
        self.x = x
        self.y = y
        self.bit_rate = bit_rate


class MusicDataConverter(object):

    def __init__(self):
        config = context.config
        self._normalize_value = config.normalize_value
        self._sampling_frequency = config.sampling_frequency
        self._block_size = config.block_size
        self._fft_enable = config.fft_enable
        self._config = config

    def save_music(self, music_file_path, generated_sequence):
        if self._fft_enable:
            time_blocks = self._fft_blocks_to_time_blocks(generated_sequence)
        else:
            time_blocks = generated_sequence
        music_data = np.concatenate(time_blocks)
        self._write_np_as_wav(music_data, music_file_path)

    def convert_mp3_to_wav(self, org_mp3_file_path, mono_mp3_file_path, wave_file_path):
        sample_freq_str = "{0:.1f}".format(float(self._sampling_frequency) / 1000.0)
        # TODO monaural or stereo どちらが良いか確認する
        cmd = 'lame -a -m m {0} {1}'.format(quote(org_mp3_file_path), quote(mono_mp3_file_path))
        # logger.info("lame monaural cmd: {}".format(cmd))
        logger.info("{}".format(cmd))
        cmds = cmd.split(" ")
        # subprocess.call(cmds, shell=True)
        cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(mono_mp3_file_path),
                                                            quote(wave_file_path),
                                                            sample_freq_str)
        # lame --decode /path/to/mp3 /path/to/wave --resample 44.1
        # logger.info("lame decode cmd: {}".format(cmd))
        logger.info("{}".format(cmd))
        cmds = cmd.split(" ")
        # subprocess.call(cmds, shell=True)

    def load_training_data(self, file_path) -> TrainingData:
        music_data, bit_rate = self._read_wav_as_np(file_path)
        x_t = self._divide_music_data(music_data, self._block_size)
        y_t = self._shift(x_t)
        if self._fft_enable:
            x_t = self._time_blocks_to_fft_blocks(x_t)
            y_t = self._time_blocks_to_fft_blocks(y_t)
        training_data = TrainingData(x_t, y_t, bit_rate)
        return training_data

    def _shift(self, data):
        shifted = data[1:]
        shifted.append(np.zeros(self._block_size))
        return shifted

    def _read_wav_as_np(self, filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / self._normalize_value
        # np_arr = np.array(np_arr)
        return np_arr, data[0]

    def _write_np_as_wav(self, music_data, music_file_path):
        music_data = music_data * self._normalize_value
        music_data = music_data.astype('int16')
        wav.write(music_file_path, self._sampling_frequency, music_data)

    @staticmethod
    def _divide_music_data(music_data: ndarray, block_size):
        block_lists = []
        total_sample_number = music_data.shape[0]
        current_sample_number = 0
        while current_sample_number < total_sample_number:
            block = music_data[current_sample_number:current_sample_number + block_size]
            if block.shape[0] < block_size:
                # TODO ステレオ対応（ステレオの場合、[][][2列]になっているので (block_size - shape[0], 2)となる）
                padding = np.zeros((block_size - block.shape[0],))
                # padding = np.zeros((block_size - block.shape[0], 2))
                block = np.concatenate((block, padding))
            block_lists.append(block)
            current_sample_number += block_size
        return block_lists

    @staticmethod
    def _time_blocks_to_fft_blocks(blocks_time_domain):
        # TODO ハミング窓適用する
        fft_blocks = []
        for block in blocks_time_domain:
            fft_block = np.fft.fft(block)
            new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
            fft_blocks.append(new_block)
        return fft_blocks

    @staticmethod
    def _fft_blocks_to_time_blocks(blocks_ft_domain):
        # TODO ハミング窓適用する
        time_blocks = []
        for block in blocks_ft_domain:
            num_elems = block.shape[0] / 2
            real_chunk = block[0:num_elems]
            imag_chunk = block[num_elems:]
            new_block = real_chunk + 1.0j * imag_chunk
            time_block = np.fft.ifft(new_block)
            time_blocks.append(time_block)
        return time_blocks
