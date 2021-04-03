import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
import SSDObjectDetection.custom_layers.L2Normalization as L2Normalization
import SSDObjectDetection.custom_layers.DefaultBoxes as DefaultBoxes
import SSDObjectDetection.custom_layers.DecodeSSDPredictions as DecodeSSDPredictions
from utils.ssd_utils import get_number_default_boxes
