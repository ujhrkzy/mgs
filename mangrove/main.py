# -*- coding: utf-8 -*-
from app.application_context_loader import ApplicationContextLoader
from training_data_generator import TrainingDataGenerator
from conductor import Conductor

__author__ = "ujihirokazuya"
__date__ = "2017/05/15"


class Main():

    def __init__(self):
        ApplicationContextLoader().load()

    @staticmethod
    def generate_training_data():
        generator = TrainingDataGenerator()
        # generator.create_wave_files()
        generator.generate()

    @staticmethod
    def train():
        conductor = Conductor()
        conductor.train()

    @staticmethod
    def generate_music():
        conductor = Conductor()
        conductor.generate_music()


if __name__ == "__main__":
    main = Main()
    # main.generate_training_data()
    # main.train()
    main.generate_music()
