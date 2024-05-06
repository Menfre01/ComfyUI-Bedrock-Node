import sys, os

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from .bedrock import Bedrock

NODE_CLASS_MAPPINGS = {
    "Bedrock": Bedrock,
}