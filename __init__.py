import sys, os

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from .bedrock import Bedrock
module_root_directory = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {
    "Bedrock": Bedrock,
}