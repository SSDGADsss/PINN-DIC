# src/model.py
import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
torch.manual_seed(123)
# CUDA
# if gpu is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available")
else:
    device = torch.device('cpu')
    print("Only cpu is available")
    
torch.backends.cudnn.benchmark = True
epsilon = torch.finfo(torch.float32).eps

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = []
        for i in range(self.depth):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

class PhysicsInformedNN1:
    def __init__(self, ref_gray_matrix, def_gray_matrix, ROI, layers, IX, IY):
        self.Iref = torch.tensor(ref_gray_matrix, dtype=torch.float32).to(device)
        self.Idef = torch.tensor(def_gray_matrix, dtype=torch.float32).to(device)
        self.IX = torch.tensor(IX, dtype=torch.float32).to(device)
        self.IY = torch.tensor(IY, dtype=torch.float32).to(device)
        self.XY = torch.stack((self.IX, self.IY), dim=2).unsqueeze(0)
        Ix = torch.flatten(self.IX)
        Iy = torch.flatten(self.IY)
        self.Ixy = torch.cat((Ix.unsqueeze(1), Iy.unsqueeze(1)), dim=1)
        self.layers = layers
        self.dnn = DNN(self.layers).to(device)
        self.dnn.initialize_weights()
        self.epoch = 0
        self.loss_list = []
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), lr=1, max_iter=2000, max_eval=2000,
            history_size=50, tolerance_grad=1e-05, 
            tolerance_change=0.5 * np.finfo(float).eps,
            line_search_fn="strong_wolfe")
        
        self.optimizer_adam = torch.optim.Adam(
            self.dnn.parameters(), lr=0.001,  eps=1e-8, weight_decay=0.001
        )

    def loss_fn1(self):
        self.optimizer.zero_grad()
        UV = self.dnn(self.Ixy)
        target_height = self.Idef.shape[0]
        target_width = self.Idef.shape[1]
        U = UV[:,0].reshape(target_height,target_width)
        V = UV[:,1].reshape(target_height,target_width)
        displacement_field = torch.stack((U, V), dim=0).unsqueeze(0)
        u = displacement_field[0,0]/target_width
        v = displacement_field[0,1]/target_height
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        X_new = self.XY + uv_displacement
        new_Idef = F.grid_sample(self.Iref.view(1, 1, target_height, target_width), 
                                 X_new.view(1, target_height, target_width, 2), 
                                 mode='bilinear', align_corners=True)
        abs_error = (new_Idef[0, 0] - self.Idef)**2
        abs_error = torch.log(1+abs_error)
        loss = torch.sum(abs_error)/(abs_error.shape[0]*abs_error.shape[1])
        loss.backward()
        self.loss_list.append(loss.item())
        self.epoch = self.epoch+1
        if self.epoch%100 == 1:   
            print(f"Epoch [{self.epoch}], Loss: {loss.item()}")
        return loss

    def loss_fn2(self):
        self.optimizer.zero_grad()
        UV = self.dnn(self.Ixy)
        target_height = self.Idef.shape[0]
        target_width = self.Idef.shape[1]
        U = UV[:,0].reshape(target_height,target_width)
        V = UV[:,1].reshape(target_height,target_width)
        displacement_field = torch.stack((U, V), dim=0).unsqueeze(0)
        u = displacement_field[0,0]/target_width
        v = displacement_field[0,1]/target_height
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        X_new = self.XY + uv_displacement
        new_Idef = F.grid_sample(self.Iref.view(1, 1, target_height, target_width), 
                                 X_new.view(1, target_height, target_width, 2), 
                                 mode='bilinear', align_corners=True)
        abs_error = (new_Idef[0, 0] - self.Idef)**2
        loss = torch.sum(abs_error)/(abs_error.shape[0]*abs_error.shape[1])
        loss.backward()
        self.loss_list.append(loss.item())
        self.epoch = self.epoch+1
        if self.epoch%100 == 1:   
            print(f"Epoch [{self.epoch}], Loss: {loss.item()}")
        return loss

    def train(self, flag):
        self.dnn.train()
        if flag==1:
            self.optimizer.step(self.loss_fn1)
        else:
            self.optimizer.step(self.loss_fn2)

    def train_adam(self, flag, epoch):
        self.dnn.train()
        if flag==1:
            for iter in range(epoch):
                loss = self.loss_fn1()
                self.optimizer_adam.step()
        else:
            for iter in range(epoch):
                loss = self.loss_fn2()
                self.optimizer_adam.step()
    
    def predict(self, X, Y):
        self.dnn.eval()
        target_height = X.shape[0]
        target_width = X.shape[1]
        IX = torch.tensor(X, dtype=torch.float32).to(device)
        IY = torch.tensor(Y, dtype=torch.float32).to(device)
        Ix = torch.flatten(IX)
        Iy = torch.flatten(IY)
        Ixy = torch.cat((Ix.unsqueeze(1), Iy.unsqueeze(1)), dim=1)
        U = self.dnn(Ixy)
        u = U[:,0].reshape(target_height,target_width)
        u = u.cpu().detach().numpy()
        v = U[:,1].reshape(target_height,target_width)
        v = v.cpu().detach().numpy()
        return u, v
