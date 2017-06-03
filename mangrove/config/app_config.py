# -*- coding: utf-8 -*-
import os.path
from resources import path

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


def _create_files(directory, extension, file_names):
    files = [os.path.join(directory, file + extension) for file in file_names]
    return files


class AppConfig(object):

    # wave file config ----------------------------
    sampling_frequency = 44100
    # length of clips for training. Defined in seconds
    clip_length = 10
    # block sizes used for training - this defines the size of our input state
    block_size = int(sampling_frequency / 4)
    # Used later for zero-padding song sequences
    max_sequence_length = int(round((sampling_frequency * clip_length) / block_size))
    fft_enable = False
    # normalize 16-bit input to [-1, 1] range
    # 2 ^ 15 - 1
    normalize_value = 32767.0
    # data shape = (num_examples, max_seq_len, num_dims_out)
    output_sequence_length = 100

    _mp3_extension = ".mp3"
    _wave_extension = ".wav"
    _org_mp3_file_directory = os.path.join(path, "train/mp3/org")
    _mono_mp3_file_directory = os.path.join(path, "train/mp3/mono")
    _wave_file_directory = os.path.join(path, "train/wave")
    _music_files = ["Test"]
    org_mp3_files = _create_files(_org_mp3_file_directory, _mp3_extension, _music_files)
    mono_mp3_files = _create_files(_mono_mp3_file_directory, "_mono" + _mp3_extension, _music_files)
    wave_files = _create_files(_wave_file_directory, _wave_extension, _music_files)

    _training_data_base_path = os.path.join(path, "train/tensor")
    x_npy = os.path.join(_training_data_base_path, "x.npy")
    y_npy = os.path.join(_training_data_base_path, "y.npy")
    x_mean_npy = os.path.join(_training_data_base_path, "x_mean.npy")
    x_std_npy = os.path.join(_training_data_base_path, "x_std.npy")

    x_mean_csv = os.path.join(_training_data_base_path, "x_mean.csv")
    x_std_csv = os.path.join(_training_data_base_path, "x_std.csv")

    # neural network config ------------------------
    # Number of hidden dimensions.
    # For best results, this should be >= freq_space_dims.
    hidden_dimension_size = 1024
    recurrent_unit_size = 3

    weight_file_directory = os.path.join(path, "results/weights")
    output_music_file_directory = os.path.join(path, "results/music")
    epoch_size = 50
    nb_epoch = 25
    batch_size = 5
    seed_length = 1

    def __init__(self):
        pass

