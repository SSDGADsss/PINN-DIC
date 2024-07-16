# src/train.py
import matplotlib.pyplot as plt
from utils.utils import zero_to_nan

def model_train(model, IX, IY, pretrain_epoch, train_epoch, gray1, gray2, new_lr=0.0001):
    model.gray_pre_train = gray1  # 预训练提前停止的阈值
    model.gray_train = gray2      # 正式训练提前停止的阈值
    if pretrain_epoch[0] != 0:
        model.train_adam(1, pretrain_epoch[0])
    if pretrain_epoch[1] != 0:
        model.train(1, pretrain_epoch[1])
    print('****************over****************')
    u,v = model.predict()
    u = zero_to_nan(u); v = zero_to_nan(v)
    plt.figure(dpi=200)
    plt.subplot(2, 2, 1)
    plt.imshow(u, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.title("Pre-train: u predicted")
    plt.subplot(2, 2, 2)
    plt.imshow(v, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.title("Pre-train: v predicted")
    
    if train_epoch[0] != 0:
        model.optimizer_adam.param_groups[0]['lr'] = new_lr
        model.train_adam(2, train_epoch[0])
    if train_epoch[1] != 0:
        model.train(2, train_epoch[1])
    u1,v1 = model.predict()
    u1 = zero_to_nan(u1); v1 = zero_to_nan(v1)
    plt.subplot(2, 2, 3)
    plt.imshow(u1, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.title("Train: u predicted")
    plt.subplot(2, 2, 4)
    plt.imshow(v1, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.title("Train: v predicted")