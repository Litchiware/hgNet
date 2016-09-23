class _const:
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError, "Can't change const %s" %name
        if not name.isupper():
            raise self.ConstCaseError, 'const name "%s" is not all uppercase' %name
        self.__dict__[name] = value

import sys
sys.modules[__name__] = _const()

import config
import os

config.SCALE_REF = 200
config.INP_RES = 256
config.INP_CHA = 3
config.OUT_RES = 64
config.OUT_CHA = 16
config.QUEUE_SIZE = 50
config.DATA_DIR = "/home/llm/Datasets/MPII"
config.IMG_DIR = os.path.join(config.DATA_DIR, "images")
config.ANNOT_DIR = os.path.join(config.DATA_DIR, "annots")
config.NUM_TRAIN = 22246
config.NUM_VALID = 2958
config.SCALE_FACTOR = 0.25
config.FLIP_CHANCE = 0.5
config.PART_MATCH = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
config.SIGMA = 1
config.TRAIN_BATCH_SIZE = 14
config.VALID_BATCH_SIZE = 29
config.WORK_DIR = "/tmp/hgnet"
config.N_FEATS = 256
config.N_STACK = 1
config.SEED = 0
config.N_EPOCHS = 100
