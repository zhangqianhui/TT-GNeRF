import torch
import torchvision.transforms.functional as VF
from torchvision import models as torch_models
from face_parsing.model import BiSeNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_SIZE = 224
ATTRIBUTE_SIZE = 224

class TransferModel(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super(TransferModel, self).__init__()
        self.backbone = backbone

        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return features

class AttributeClassifier(torch.nn.Module):
    def __init__(
        self,
        path='./checkpoints/attr-classifier-resnet50-gs/classifier_check_last.pt',
        preprocess=True,
        device=None
    ):
        super(AttributeClassifier, self).__init__()
        self.preprocess = preprocess

        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.load_state_dict(torch.load('/nfs/data_todi/jzhang/code/eg3d/eg3d/pretrained_model/79999_iter.pth'))
        for param in self.net.parameters():
            param.requires_grad = False

        self.net.eval()

        self.attr = TransferModel(backbone=torch_models.resnet50(pretrained=True), num_classes=40)
        checkpoint = torch.load('./checkpoints/attr-classifier-resnet50-gs/classifier_check_last.pt', map_location=torch.device('cpu'))
        self.attr.load_state_dict(checkpoint)
        self.attr.eval()
        self.attr.to(device)

        for module in [self.attr]:
            for param in module.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # x in [-1 1]
        x = VF.center_crop(VF.resize(x, ATTRIBUTE_SIZE), ATTRIBUTE_SIZE)
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        if self.preprocess:
            x = self._preprocess(x)

        feat = self.attr(x)
        return torch.sigmoid(feat)
