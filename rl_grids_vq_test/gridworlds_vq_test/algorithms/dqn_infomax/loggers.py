import argparse
import json
import os
import random
import json
import time
import numpy as np
import logging

import nested_lookup
from tensorboardX import SummaryWriter
from tensorboardX.event_file_writer import EventsWriter
import types


class SummaryWriter_X(object):
    def __init__(self, log_dir, use_TFX):
        self.logger = SummaryWriter(log_dir)
        self.use_TFX = use_TFX

    def log(self, name, value, step):
        self.logger.add_scalar(name, value, step)
        if self.use_TFX:
            self._flush()

    def _flush(self):
        try:
            path = self.logger.file_writer.event_writer._ev_writer._py_recordio_writer.path
            self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer.flush()
            while True:
                if self.logger.file_writer.event_writer._event_queue.empty():
                    break
                time.sleep(0.1)  # Increased from 0.1 -> X s
            # self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer.close()
            self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer = open(path, 'ab')
        except:
            pass


class Logger(object):
    def __init__(self, log_dir, use_TFX,params=None,comet_experiment=None,disable_local=False):
        self.log_dir = log_dir
        self.use_TFX = use_TFX
        self.disable_local = disable_local

        if not self.disable_local:
            self.text_logger = TextLogger(log_dir)
            self.csv_logger = CSVLogger(log_dir)
            # tb_path = self.log_dir + '_tensorboard'
            # self.tensorboard_logger = SummaryWriter_X(tb_path, self.use_TFX)
        self.comet_logger = comet_experiment

        if not self.disable_local:
            if params is not None:
                with open(log_dir+'_params.json', 'w+') as f:
                    json.dump(params,f)
                    f.flush()
                    os.fsync(f.fileno())

    def log(self, s):
        if type(s) == str:
            if not self.disable_local:
                self.text_logger.log(s)
        elif type(s) == dict:
            name, value, step = s['name'], s['value'], s['step']
            if not self.disable_local:
                # self.tensorboard_logger.log(name, value, step)
                self.csv_logger.log(name, value, step)

            if self.comet_logger:
                self.comet_logger.log_metric(name, value, step=step)


class TextLogger(object):
    def __init__(self, log_dir):
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        self.logger = logging.getLogger('custom')
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fileHandler = logging.FileHandler('%s_log.log' % (log_dir))
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.INFO)
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def log(self, s):
        self.logger.info(s)
        for h in self.logger.handlers:
            h.flush()

class CSVLogger(object):
    def __init__(self, log_dir):
        self.log_file = log_dir + '_log.csv'
        with open(self.log_file,'w') as fh:
            fh.write('Key\tValue\tStep\t')

    def log(self, name,value,step):
        with open(self.log_file,'a') as fh:
            fh.write('%s\t%f\t%d\n'%(name,value,step))