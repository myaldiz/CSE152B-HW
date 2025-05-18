import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np



def cosine_sim(x1, x2, dim=1, eps=1e-8):
    """
    TODO: Implement cosine similarity
    Hint: add eps to avoid zero division
    """
    pass

class CustomLoss(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features=512, classnum=10574, s=30.0, m=0.40):
        super(CustomLoss, self).__init__()
        self.in_features = in_features
        out_features = classnum
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # TODO: Implement cosFace loss:
        output = None
        
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, label.squeeze())

        # get accuracy
        _, predictedLabel = torch.max(cosine, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(label).cpu().sum().item() ) / float(label.size(0) )

        return loss, accuracy


class faceNet(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super().__init__()
        self.classnum = classnum
        self.feature = feature

        # ---------- stage-1  (1 basic residual block) ----------
        self.conv1_1 = nn.Conv2d(3,  64, 3, 2, 1)  # → B×64×56×48
        self.prelu1_1 = nn.PReLU(64)

        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu1_3 = nn.PReLU(64)

        # ---------- stage-2  (2 basic residual blocks) ----------
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)      # → B×128×28×24
        self.prelu2_1 = nn.PReLU(128)

        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.prelu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.prelu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.prelu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.prelu2_5 = nn.PReLU(128)

        # ---------- stage-3  (4 basic residual blocks) ----------
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)     # → B×256×14×12
        self.prelu3_1 = nn.PReLU(256)

        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3_9 = nn.PReLU(256)

        # ---------- stage-4  (1 basic residual block) ----------
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)     # → B×512×7×6
        self.prelu4_1 = nn.PReLU(512)

        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.prelu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.prelu4_3 = nn.PReLU(512)

        # ---------- embedding ----------
        self.fc = nn.Linear(512 * 7 * 6, 512)
        self._init_weights()

    # -----------------------------------------------------------
    # Forward pass (mirrors residual additions of original Block)
    # -----------------------------------------------------------
    def forward(self, x):
        # stage-1
        x = self.prelu1_1(self.conv1_1(x))
        x = x + self.prelu1_3(self.conv1_3(self.prelu1_2(self.conv1_2(x))))

        # stage-2
        x = self.prelu2_1(self.conv2_1(x))
        x = x + self.prelu2_3(self.conv2_3(self.prelu2_2(self.conv2_2(x))))
        x = x + self.prelu2_5(self.conv2_5(self.prelu2_4(self.conv2_4(x))))

        # stage-3
        x = self.prelu3_1(self.conv3_1(x))
        x = x + self.prelu3_3(self.conv3_3(self.prelu3_2(self.conv3_2(x))))
        x = x + self.prelu3_5(self.conv3_5(self.prelu3_4(self.conv3_4(x))))
        x = x + self.prelu3_7(self.conv3_7(self.prelu3_6(self.conv3_6(x))))
        x = x + self.prelu3_9(self.conv3_9(self.prelu3_8(self.conv3_8(x))))

        # stage-4
        x = self.prelu4_1(self.conv4_1(x))
        x = x + self.prelu4_3(self.conv4_3(self.prelu4_2(self.conv4_2(x))))

        # embedding
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    # -----------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if m.bias is not None:          # same rule as original code
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0.0, 0.01)
