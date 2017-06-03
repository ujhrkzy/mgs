# -*- coding: utf-8 -*-
import logging.config
import os

log_dir_name = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


logging_config = dict(
    version=1,
    formatters={
        'simpleFormatter': {'format': "[%(asctime)s][%(levelname)-7s][%(module)s-%(lineno)d] - %(message)s",
                            'datefmt': "%Y%m%d %H:%M:%S"}
    },
    handlers={
        'consoleHandler': {'class': 'logging.StreamHandler',
                           'formatter': 'simpleFormatter',
                           'level': logging.getLevelName(logging.DEBUG)},
        'infoHandler': {'class': 'logging.handlers.RotatingFileHandler',
                        'formatter': 'simpleFormatter',
                        'level': logging.getLevelName(logging.DEBUG),
                        'filename': log_dir_name + '/mangrove_info.log',
                        'maxBytes': 1024*1024*500,
                        'backupCount': 4,
                        'encoding': 'utf-8'}
    },
    loggers={
        'infoLog': {'handlers': ['consoleHandler', 'infoHandler'],
                    'level': logging.getLevelName(logging.DEBUG),
                    'propagate': True}
    }
)
logging.config.dictConfig(logging_config)

logger = logging.getLogger('infoLog')
