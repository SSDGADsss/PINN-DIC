from .FCNN import DNN
from .PINN import PhysicsInformedNN

# 控制使用 from src import * 时的导入内容
__all__ = ["DNN", "PhysicsInformedNN"]