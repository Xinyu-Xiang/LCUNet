import torch.nn as nn 
import torch.nn.functional as F


class a_projector(nn.Module):
    '''
    project from audio to video domain 
    '''
    def __init__(self, input_dim, output_dim):
        super(a_projector, self).__init__()
        self.linear_down = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        self.linear_remain = nn.Linear(input_dim // 2, input_dim // 2)
        self.linear_up = nn.Linear(input_dim // 2, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.linear_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_remain(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_up(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        return x

class v_projector(nn.Module):
    '''
    project from face to video domain 
    '''
    def __init__(self, input_dim, output_dim):
        super(v_projector, self).__init__()
        self.linear_down = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        self.linear_remain = nn.Linear(input_dim // 2, input_dim // 2)
        self.linear_up = nn.Linear(input_dim // 2, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.linear_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_remain(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_up(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        return x
    


class f_projector(nn.Module):
    '''
    project from face to video domain 
    '''
    def __init__(self, input_dim, output_dim):
        super(f_projector, self).__init__()
        self.linear_down = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        self.linear_remain = nn.Linear(input_dim // 2, input_dim // 2)
        self.linear_up = nn.Linear(input_dim // 2, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = x.squeeze(-1)
        x = self.linear_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_remain(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_up(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.unsqueeze(-1)
        return x


class a_projector_aux(nn.Module):
    '''
    project from face to video domain 
    '''
    def __init__(self, input_dim, output_dim):
        super(a_projector_aux, self).__init__()
        self.linear_down = nn.Linear(input_dim, input_dim // 2)
        self.bn1 = nn.BatchNorm1d(input_dim // 2)
        self.relu = nn.ReLU()
        self.linear_up = nn.Linear(input_dim // 2, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = self.linear_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_up(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = x.unsqueeze(-1)
        return x


class va_mix_proj(nn.Module):
    '''
    project from face to video domain 
    '''
    def __init__(self, input_dim, output_dim):
        super(va_mix_proj, self).__init__()
        self.linear_down = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.linear_up = nn.Linear(input_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = self.linear_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear_up(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = x.unsqueeze(-1)
        return x
    

class weight_pred2d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(weight_pred2d, self).__init__()
        self.classifier_weight = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, output_dim),
        )  # fused 逻辑值进行预估
    def forward(self, x):
        x = self.classifier_weight(x)
        x = F.softmax(x, dim=-1)
        return x
    

class weight_pred3d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(weight_pred3d, self).__init__()
        self.classifier_weight = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, output_dim),
        )  # fused 逻辑值进行预估
    def forward(self, x):
        x = self.classifier_weight(x)
        x = F.softmax(x, dim=-1)
        return x