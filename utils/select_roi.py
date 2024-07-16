import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class ROISelector:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)  
        self.image = self.original_image.copy()
        self.clone = self.image.copy()       # 储存加减后的ROI
        self.clone_temp = self.clone.copy()  # 储存新画的ROI
        self.image_flag = 0                  # 控制显示哪个画布
        self.roi = np.zeros(self.image.shape[:2], dtype=np.uint8)  # ROI mask
        self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False   # 是否正在绘制或调整
        self.resizing = False  # 是否正在调整
        self.roi_selected = False      # 是否选择了ROI
        self.selected_handle = None    # 选中的控制点
        self.handle_size = 10          # 鼠标在控制点附近10像素以内，被选中
        self.mode = 'rectangle'  # 可以是圆或是矩形
        self.operation = 'add'  # 可以是加操作或者减操作
        self.start_point = None  # 初始点
        self.rx=0; self.ry=0; self.rw=0; self.rh=0; 
        self.cx=0; self.cy=0; self.radius=0; 
        
    def find_selected_handle_circle(self, x, y, cx, cy, radius):
        """ 查找被选中的控制点 """
        if abs(x - cx) <= self.handle_size and abs(y - cy) <= self.handle_size:
            return "center"
        elif abs((x - cx)) <= self.handle_size and abs((y - (cy - radius))) <= self.handle_size:
            return "top"
        elif abs((x - cx)) <= self.handle_size and abs((y - (cy + radius))) <= self.handle_size:
            return "bottom"
        elif abs((x - (cx - radius))) <= self.handle_size and abs((y - cy)) <= self.handle_size:
            return "left"
        elif abs((x - (cx + radius))) <= self.handle_size and abs((y - cy)) <= self.handle_size:
            return "right"
        else:
            return None
        
    def find_selected_handle_rectangle(self, x, y, rect):
        """ 查找被选中的边或角 """
        rx, ry, rw, rh = rect
        if abs(x - rx) <= self.handle_size and abs(y - ry) <= self.handle_size:
            return "topleft"
        elif abs(x - (rx + rw)) <= self.handle_size and abs(y - ry) <= self.handle_size:
            return "topright"
        elif abs(x - rx) <= self.handle_size and abs(y - (ry + rh)) <= self.handle_size:
            return "bottomleft"
        elif abs(x - (rx + rw)) <= self.handle_size and abs(y - (ry + rh)) <= self.handle_size:
            return "bottomright"
        else:
            return None
        
    def update_circle(self, x, y, handle):
        """ 根据控制点调整圆 """
        if handle == "center":
            self.cx, self.cy = x, y
        elif handle == "top":
            self.radius = abs(y - self.cy)
        elif handle == "bottom":
            self.radius = abs(y - self.cy)
        elif handle == "left":
            self.radius = abs(x - self.cx)
        elif handle == "right":
            self.radius = abs(x - self.cx)
            
    def update_rectangle(self, x, y, handle):
        """ 根据边缘调整 ROI """
        if handle == "topleft":
            self.rw, self.rh = self.rw + (self.rx - x), self.rh + (self.ry - y)
            self.rx, self.ry = x, y
            # self.rw, self.rh = self.rw + (self.start_point[0] - x), self.rh + (self.start_point[1] - y)
        elif handle == "topright":
            self.rw, self.rh = x - self.rx, self.rh + (self.ry - y)
            self.ry = y
            # self.rw, self.rh = x - self.rx, self.rh + (self.start_point[1] - y)
        elif handle == "bottomleft":
            self.rw, self.rh = self.rw + (self.rx - x), y - self.ry
            self.rx = x
            # self.rw, self.rh = self.rw + (self.start_point[0] - x), y - self.ry
        elif handle == "bottomright":
            self.rw, self.rh = x - self.rx, y - self.ry
            
    def draw_circle(self):
        """ 绘制圆和控制点 """
        # 在哪个画布上？？？？？？  应该在event上展示
        cv2.circle(self.image, (self.cx, self.cy), self.radius, (0, 255, 0), 2)
        cv2.circle(self.image, (self.cx, self.cy), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx, self.cy - self.radius), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx, self.cy + self.radius), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx - self.radius, self.cy), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx + self.radius, self.cy), self.handle_size, (255, 0, 0), -1)
        
    def draw_rectangle(self):
        """ 绘制 ROI 和控制点 """
        # 在哪个画布上？？？？？？
        cv2.rectangle(self.image, (self.rx, self.ry), (self.rx + self.rw, self.ry + self.rh), (0, 255, 0), 2)
        cv2.circle(self.image, (self.rx, self.ry), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx + self.rw, self.ry), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx, self.ry + self.rh), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx + self.rw, self.ry + self.rh), self.handle_size, (255, 0, 0), -1)
        
    def display_roi(self):
        if self.image_flag == 0:
            if self.mode == 'circle':
                self.draw_circle()
            elif self.mode == 'rectangle':
                self.draw_rectangle()
        elif self.image_flag == 1:
            self.image = self.clone.copy()

    def draw_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.roi_selected:
                self.drawing = True
                self.image_flag = 0   #######
                self.start_point = (x, y)
                if self.mode == 'circle':
                    self.cx, self.cy, self.radius = self.start_point[0], self.start_point[1], 0
                elif self.mode == 'rectangle':
                    self.rx, self.ry, self.rw, self.rh = self.start_point[0], self.start_point[1], 0, 0
            else:
                if self.mode == 'circle':
                    self.selected_handle = self.find_selected_handle_circle(
                        x, y, self.cx, self.cy, self.radius)
                    if self.selected_handle:
                        self.resizing = True
                        self.start_point = (x, y)
                elif self.mode == 'rectangle':
                    self.selected_handle = self.find_selected_handle_rectangle(
                        x, y, (self.rx, self.ry, self.rw, self.rh))
                    if self.selected_handle:
                        self.resizing = True
                        self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.mode == 'circle':
                    self.radius = int(np.sqrt((self.start_point[0] - x) ** 2 + (self.start_point[1] - y) ** 2))
                elif self.mode == 'rectangle':
                    self.rx, self.ry = min(self.start_point[0], x), min(self.start_point[1], y)
                    self.rw, self.rh = abs(self.start_point[0] - x), abs(self.start_point[1] - y)
                # self.update_display(x, y)    #   这里要修改
            elif self.resizing:
                if self.mode == 'circle':
                    self.update_circle(x, y, self.selected_handle)
                elif self.mode == 'rectangle':
                    self.update_rectangle(x, y, self.selected_handle)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.roi_selected = True
            elif self.resizing:
                self.resizing = False

    def update_display(self):
        self.clone = self.clone_temp.copy()
        self.clone = cv2.addWeighted(self.clone, 1, cv2.cvtColor(self.roi, cv2.COLOR_GRAY2BGR), 0.5, 0)
        self.image_flag = 1
        
    def load_roi(self):
        self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.mode == 'rectangle':
            cv2.rectangle(self.new_roi, (self.rx, self.ry), (self.rx + self.rw, self.ry + self.rh), 255, -1)
        elif self.mode == 'circle':
            cv2.circle(self.new_roi, (self.cx, self.cy), self.radius, 255, -1)

    def apply_operation(self):
        self.load_roi()
        if self.operation == 'add':
            self.roi = cv2.bitwise_or(self.roi, self.new_roi)
        elif self.operation == 'subtract':
            self.roi = cv2.bitwise_and(self.roi, cv2.bitwise_not(self.new_roi))
        self.update_display()
        self.drawing = False   # 是否正在绘制或调整
        self.resizing = False  # 是否正在调整
        self.roi_selected = False      # 是否选择了ROI
        self.selected_handle = None    # 选中的控制点
        self.start_point = None  # 初始点
        self.rx=0; self.ry=0; self.rw=0; self.rh=0; 
        self.cx=0; self.cy=0; self.radius=0;
        

    def run(self):
        cv2.namedWindow("ROI Selector")
        cv2.setMouseCallback("ROI Selector", self.draw_roi)

        while True:
            self.image = self.original_image.copy()
            self.display_roi()
            cv2.imshow("ROI Selector", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                self.operation = 'add'
            elif key == ord('s'):
                self.operation = 'subtract'
            elif key == ord('c'):
                self.mode = 'circle'
            elif key == ord('r'):
                self.mode = 'rectangle'
            elif key == ord('f'):
                self.apply_operation()
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.roi

def ROI_bmp():
    Tk().withdraw()  # 隐藏根 Tkinter 窗口
    image_path = askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.bmp;*.jpg;*.jpeg;*.png;*.tiff;*.tif")]
    )   # 要求用户选择一个图像文件
    if not image_path:
        print("No image selected. Exiting.")
        return
    # 实例化ROISelector
    selector = ROISelector(image_path)
    final_roi = selector.run()
    # 将 ROI 保存到当前工作目录
    cv2.imwrite('final_roi.png', final_roi)
    print(f"Final ROI saved as 'final_roi.png' in the current directory.")
    # 将ROI保存到所选图像的目录中
    image_dir = os.path.dirname(image_path)
    save_path = os.path.join(image_dir, 'roi.bmp')
    cv2.imwrite(save_path, final_roi)
    print(f"Final ROI saved as 'roi.bmp' in the directory of the selected image.")

if __name__ == "__main__":
    ROI_bmp()
