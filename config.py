import yaml
import os

from objects.config import Conf

# config_yaml = yaml.safe_load(open("../conf/config.yaml"))
config_yaml = yaml.safe_load(open("./conf/config.yaml"))

configuration = Conf(**config_yaml)