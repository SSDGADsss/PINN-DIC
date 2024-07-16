# src/model.py
import torch
import torch.nn.functional as F
from .FCNN import DNN
import numpy as np
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilon = torch.finfo(torch.float32).eps


class PhysicsInformedNN:
    def __init__(self, ref_gray_matrix, def_gray_matrix, ROI, layers, IX, IY, XY_roi):
        self.Iref = torch.tensor(ref_gray_matrix, dtype=torch.float32).to(device)
        self.Idef = torch.tensor(def_gray_matrix, dtype=torch.float32).to(device)
        self.ROI = torch.tensor(ROI, dtype=torch.float32).to(device)
        self.IX = torch.tensor(IX, dtype=torch.float32).to(device)  # 2D grid
        self.IY = torch.tensor(IY, dtype=torch.float32).to(device)  # 2d grid
        self.XY = torch.stack((self.IX, self.IY), dim=2).unsqueeze(0); 
        self.XY_roi = torch.tensor(XY_roi).to(device) # n_row; 2_col
        self.Ixy = torch.zeros_like(self.XY_roi).to(device)
        self.Ixy = self.Ixy.float() 
        self.Ixy[:,0] = self.IX[self.XY_roi[:, 0], self.XY_roi[:, 1]]
        self.Ixy[:,1] = self.IY[self.XY_roi[:, 0], self.XY_roi[:, 1]]
        
        # self.U = torch.zeros_like(self.Iref, requires_grad=True).to(device)  #用于计算过程中储存结果
        # self.V = torch.zeros_like(self.Iref, requires_grad=True).to(device)  #用于计算过程中储存结果
        # self.U = torch.zeros_like(self.Iref).to(device)  #用于计算过程中储存结果
        # self.V = torch.zeros_like(self.Iref).to(device)  #用于计算过程中储存结果
        
        self.layers = layers
        self.dnn = DNN(self.layers).to(device)
        self.epoch = 0; self.gray_mse_list = []; self.gray_mae_list = []
        self.train_flag = False  # 为True时提前停止训练
        self.gray_pre_train = 5  # 预训练提前停止的阈值
        self.gray_train = 3      # 正式训练提前停止的阈值
        
        self.max_iter = 50;  self.max_eval = int(1.25 * self.max_iter)
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), lr=1, max_iter=50, max_eval=75,
            history_size=50, tolerance_grad=1e-04, 
            tolerance_change=5e-04,
            line_search_fn="strong_wolfe")
        
        self.optimizer_adam = torch.optim.Adam(
            self.dnn.parameters(), lr=0.001,  eps=1e-8)
        
    def fill_U_with_roi(self, UV):
        coords = self.XY_roi
        U = torch.zeros_like(self.Iref).to(device)
        V = torch.zeros_like(self.Iref).to(device)
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        U[y_coords, x_coords] = UV[:, 0]  # Assuming 0 is for u
        V[y_coords, x_coords] = UV[:, 1]  # Assuming 1 is for v
        return U, V
    
    def interp_IPD(self, U, V):
        target_height = self.Idef.shape[0]; target_width = self.Idef.shape[1]
        u = -U/(target_width/2); v = -V/(target_height/2)
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        # print(uv_displacement.shape); print(self.XY.shape);
        X_new = self.XY + uv_displacement
        new_Idef = F.grid_sample(self.Iref.view(1, 1, target_height, target_width), 
                                 X_new.view(1, target_height, target_width, 2), 
                                 mode='bilinear', align_corners=True)
        return new_Idef
    
    def update_list(self, new_Idef):
        abs_error = (new_Idef[0, 0] - self.Idef)**2 * self.ROI
        mse = torch.sum(abs_error)/self.XY_roi.shape[0]
        self.gray_mse_list.append(mse.item())
        absolute_error = torch.abs(new_Idef[0, 0] - self.Idef) * self.ROI
        mae = torch.sum(absolute_error)/self.XY_roi.shape[0]
        self.gray_mae_list.append(mae.item())
        return abs_error, mse, mae
        
    def loss_fn1(self):
        self.optimizer.zero_grad()
        UV = self.dnn(self.Ixy)
        U, V = self.fill_U_with_roi(UV)
        new_Idef = self.interp_IPD(U, V)
        abs_error, mse, mae = self.update_list(new_Idef)
        abs_error = torch.log(1+abs_error)
        loss = torch.sum(abs_error)/self.XY_roi.shape[0]
        loss.backward()
        if self.gray_pre_train > mae:
            self.train_flag = True
        self.epoch = self.epoch+1
        if self.epoch%100 == 1:
            print(f"Epoch [{self.epoch}], Loss: {loss.item():.4f}, MAE: {mae.item():.4f}")
        return loss

    def loss_fn2(self):
        self.optimizer.zero_grad()
        UV = self.dnn(self.Ixy)
        U, V = self.fill_U_with_roi(UV)
        new_Idef = self.interp_IPD(U, V)
        abs_error, mse, mae = self.update_list(new_Idef)
        loss = mse
        loss.backward()
        if self.gray_train > mae:
            self.train_flag = True
        self.epoch = self.epoch+1
        if self.epoch%100 == 1:   
            print(f"Epoch [{self.epoch}], Loss: {loss.item():.4f}, MAE: {mae.item():.4f}")
        return loss

    def train(self, flag, epoch1):
        self.dnn.train()
        epoch = epoch1//self.max_iter
        if flag==1:
            for iter in range(epoch):
                self.optimizer.step(self.loss_fn1)
                if self.train_flag:
                    self.train_flag = False
                    break
        else:
            for iter in range(epoch):
                self.optimizer.step(self.loss_fn2)
                if self.train_flag:
                    self.train_flag = False
                    break

    def train_adam(self, flag, epoch):
        self.dnn.train()
        if flag==1:
            for iter in range(epoch):
                loss = self.loss_fn1()
                if self.train_flag:
                    self.train_flag = False
                    break
                self.optimizer_adam.step()
        else:
            for iter in range(epoch):
                loss = self.loss_fn2()
                self.optimizer_adam.step()
                if self.train_flag:
                    self.train_flag = False
                    break
    
    def predict(self):
        self.dnn.eval()
        UV = self.dnn(self.Ixy)
        u,v = self.fill_U_with_roi(UV)
        u = u.cpu().detach().numpy()
        v = v.cpu().detach().numpy()
        return u, v


