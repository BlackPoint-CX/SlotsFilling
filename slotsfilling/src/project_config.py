import ConfigParser
import os

config_parser = ConfigParser()
config_parser.read('./basic.conf')

PROJECT_PATH = config_parser.read('PROJECT_CONFIG', 'PROJECT_PATH')
DATA_DIR = os.path.join(PROJECT_PATH, 'slotsfilling/data')
SRC_DIR = os.path.join(PROJECT_PATH, 'slotsfilling/src')
