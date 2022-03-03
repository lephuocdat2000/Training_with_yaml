from torch import batch_norm, nn 
import torchvision.models as models


class EfficientNetb7(nn.Module):
    def __init__(self, num_classes, pretrained, features_fixed=True):
        super(EfficientNetb7, self).__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        self.model.features.requires_grad_(not features_fixed)
        self.model.classifier[-1] = nn.Linear(in_features=2560, out_features=1024)
        self.model = nn.Sequential(self.model,nn.ReLU(),nn.Linear(1024,2))
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        outputs = self.model(x)
        outputs = self.softmax(outputs)
        return outputs

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
