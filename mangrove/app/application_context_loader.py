# -*- coding: utf-8 -*-
from config.app_config import AppConfig
from app.application_context import context


class ApplicationContextLoader(object):

    def __init__(self):
        pass

    @staticmethod
    def load(config=None):
        if config is None:
            config = AppConfig()
        context.add(AppConfig, config)


