import md5
import pdb
import yaml
CONFIG = yaml.load(open("config.yaml", "rb"))
import sys
sys.path.append(CONFIG["svm"]["path"])

import re
import os
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    level=logging.DEBUG)


def get_file_path_from_label(label):
    regex_match = re.match('(.*)_(.*)', label)
    assert len(regex_match.groups()) == 2
    assert regex_match.groups()[0][0] == 'n'
    assert regex_match.groups()[1].isdigit()
    class_dir_name = regex_match.groups()[0]
    file_path = os.path.join(*[
        CONFIG["database_config"]["base_directory_name"],
        CONFIG["database_config"]["image_base_dir_rel_path"],
        class_dir_name, "images", label])

    return file_path

