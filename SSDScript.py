import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from custom_layers import L2Normalization, DefaultBoxes, DecodeSSDPredictions
from utils.ssd_utils import get_number_default_boxes
