import torch
import os

# The paths of our folders
train = os.path.join("dataset", "train")
test = os.path.join("dataset", "test")
output = "output"
testIm_paths = os.path.sep.join([output, "test_paths.txt"])
HHIm_paths = os.path.join(output, "testHH")

train_image = os.path.join(train, "images")
train_label = os.path.join(train, "labels_1D")
test_image = os.path.join(test, "images")
test_label = os.path.join(test, "labels_1D")

dbg_train_image = os.path.join(train, "images_debug")
dbg_train_label = os.path.join(train, "labels_1D_debug")
dbg_test_image = os.path.join(test, "images_debug")
dbg_test_label = os.path.join(test, "labels_1D_debug")

#Output paths
dbg_model = os.path.join(output, "7", "unet_oil.pth")
model = os.path.join(output, "unet_oil.pth")
plot_path = os.path.sep.join([output, "plot.png"])

validationMetric_path = os.path.sep.join([output, "metric_plotval.png"])
trainMetric_path = os.path.sep.join([output, "metric_plottra.png"])
validationAccuracy_path = os.path.sep.join([output, "accuracy_plotval.png"])
trainAccuracy_path = os.path.sep.join([output, "accuracy_plottra.png"])
trainConf_path = os.path.sep.join([output, "conf_plottra.png"])
validationConf_path = os.path.sep.join([output, "conf_plotvalid.png"])

# What type of hardware it trains on
device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we are pinning memory during the data loading
pin_memory = True if device == "cuda" else False

# Define the number of channels
num_classes = 5

# Parameters
init_LR = 5*10e-5
num_epochs = 600
batch_size = 12
patch_size = (320, 320)

# Image dimensions
IM_width = 1250
IM_height = 650