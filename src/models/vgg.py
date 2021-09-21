from torch import nn
from utils.ste import LinearQuantized, Conv2dQuantized

class VGG_small(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, input_channel=3, n_classes=1000):
        super(VGG_small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class VGG_Quantized(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, input_channel=3, n_classes=1000):
        super(VGG_Quantized, self).__init__()
        self.features = nn.Sequential(
            Conv2dQuantized(in_channels=input_channel, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dQuantized(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            Conv2dQuantized(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            LinearQuantized(512 * 4 * 4, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            LinearQuantized(1024, n_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
