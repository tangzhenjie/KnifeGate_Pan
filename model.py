import torch.nn as nn
import networks

class Resnet50FCN(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50FCN, self).__init__()

        # Encoder Network
        self.Encoder = networks.Encoder(is_restore_from_imagenet=True, resnet_weight_path="./resnetweight/")
        # Semantic Segmentation Branch
        self.SegClassifier = networks.Classifier_Module([6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, img):
        n, c, h, w = img.shape
        # forward the data
        backbone = self.Encoder(img)
        # get the high-resolution image, the segmentation map and two features for FA
        seg_pre = self.SegClassifier(backbone)
        seg_pre = nn.functional.interpolate(seg_pre, size=(h, w), mode='bilinear')
        return seg_pre