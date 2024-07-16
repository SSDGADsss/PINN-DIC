# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:27:23 2024

@author: 28622
"""

import cv2
ref_image_path = "../speckle_image/real/circle/nuclear_ring_ref.bmp"
roi_path = "../speckle_image/real/circle/roi.bmp"
# ref_image_path = "../speckle_image/real/strip/LBD__00.bmp"
# roi_path = "../speckle_image/real/strip/roi.bmp"
ref = cv2.imread(ref_image_path)
roi = cv2.imread(roi_path)
blended_image = cv2.addWeighted(ref, 1, roi, 0.5, 0)
output_filename = 'blended_image.png'  # 您可以自定义输出文件名
cv2.imwrite(output_filename, blended_image)











# import cv2
# import numpy as np

# if __name__ == '__main__' :
#     # Read image
#     img = cv2.imread("../speckle_image/real/circle/nuclear_ring_ref.bmp")
#     cv2.namedWindow("Image",2)
#     roi = cv2.selectROI("Image", img, False, False)

#     ## Display the roi
#     if roi is not None:
#         x,y,w,h = roi
#         mask = np.zeros_like(img, np.uint8)
#         cv2.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)
#         masked = cv2.bitwise_and(img, mask )
#         cv2.imshow("mask", mask)
#         cv2.imshow("ROI", masked)

#     cv2.waitKey()
#####   ../speckle_image/real/circle/nuclear_ring_ref.bmp






# import cv2
# import numpy as np

# # 初始化变量
# ix, iy = -1, -1  # 初始点
# drawing = False  # 是否正在绘制或调整
# resizing = False  # 是否正在调整
# roi_selected = False  # 是否选择了ROI
# selected_handle = None  # 选中的控制点

# # 初始化圆的参数
# cx, cy, radius = 0, 0, 0

# # 定义控制点的检测区域
# handle_size = 10

# def is_point_in_circle(x, y, cx, cy, radius):
#     """ 检查点是否在圆内 """
#     return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

# def find_selected_handle(x, y, cx, cy, radius):
#     """ 查找被选中的控制点 """
#     if abs(x - cx) <= handle_size and abs(y - cy) <= handle_size:
#         return "center"
#     elif abs((x - cx)) <= handle_size and abs((y - (cy - radius))) <= handle_size:
#         return "top"
#     elif abs((x - cx)) <= handle_size and abs((y - (cy + radius))) <= handle_size:
#         return "bottom"
#     elif abs((x - (cx - radius))) <= handle_size and abs((y - cy)) <= handle_size:
#         return "left"
#     elif abs((x - (cx + radius))) <= handle_size and abs((y - cy)) <= handle_size:
#         return "right"
#     else:
#         return None

# def draw_circle(img, cx, cy, radius):
#     """ 绘制圆和控制点 """
#     cv2.circle(img, (cx, cy), radius, (0, 255, 0), 2)
#     cv2.circle(img, (cx, cy), handle_size, (255, 0, 0), -1)
#     cv2.circle(img, (cx, cy - radius), handle_size, (255, 0, 0), -1)
#     cv2.circle(img, (cx, cy + radius), handle_size, (255, 0, 0), -1)
#     cv2.circle(img, (cx - radius, cy), handle_size, (255, 0, 0), -1)
#     cv2.circle(img, (cx + radius, cy), handle_size, (255, 0, 0), -1)

# def update_circle(x, y, handle, cx, cy, radius):
#     """ 根据控制点调整圆 """
#     if handle == "center":
#         cx, cy = x, y
#     elif handle == "top":
#         radius = abs(y - cy)
#     elif handle == "bottom":
#         radius = abs(y - cy)
#     elif handle == "left":
#         radius = abs(x - cx)
#     elif handle == "right":
#         radius = abs(x - cx)
#     return cx, cy, radius

# def mouse_callback(event, x, y, flags, param):
#     global ix, iy, drawing, resizing, roi_selected, selected_handle, cx, cy, radius
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not roi_selected:
#             ix, iy = x, y
#             cx, cy, radius = ix, iy, 0
#             drawing = True
#         else:
#             selected_handle = find_selected_handle(x, y, cx, cy, radius)
#             if selected_handle:
#                 resizing = True
#                 ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             radius = int(np.sqrt((ix - x) ** 2 + (iy - y) ** 2))
#         elif resizing:
#             cx, cy, radius = update_circle(x, y, selected_handle, cx, cy, radius)

#     elif event == cv2.EVENT_LBUTTONUP:
#         if drawing:
#             drawing = False
#             roi_selected = True
#         elif resizing:
#             resizing = False

# # 创建窗口
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', mouse_callback)

# # 载入图像
# img = cv2.imread('../speckle_image/real/circle/nuclear_ring_ref.bmp')
# orig = img.copy()

# while True:
#     img = orig.copy()
#     # 画出圆形 ROI
#     draw_circle(img, cx, cy, radius)
#     cv2.imshow('image', img)
    
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# # 输出最终的圆形 ROI
# print("Final Circle ROI: Center =", (cx, cy), "Radius =", radius)























# import cv2
# import numpy as np

# # 初始化变量
# ix, iy = -1, -1 # 初始点
# drawing = False # 是否正在绘制或调整
# resizing = False # 是否正在调整
# roi_selected = False # 是否选择了ROI
# selected_edge = None # 选中的边缘

# # 定义四个角点和四条边的检测区域
# corner_size = 10
# edge_size = 5

# def is_point_in_rect(x, y, rect):
#     """ 检查点是否在矩形内部 """
#     rx, ry, rw, rh = rect
#     return rx <= x <= rx + rw and ry <= y <= ry + rh

# def find_selected_edge(x, y, rect):
#     """ 查找被选中的边或角 """
#     rx, ry, rw, rh = rect
    
#     if abs(x - rx) <= corner_size and abs(y - ry) <= corner_size:
#         return "topleft"
#     elif abs(x - (rx + rw)) <= corner_size and abs(y - ry) <= corner_size:
#         return "topright"
#     elif abs(x - rx) <= corner_size and abs(y - (ry + rh)) <= corner_size:
#         return "bottomleft"
#     elif abs(x - (rx + rw)) <= corner_size and abs(y - (ry + rh)) <= corner_size:
#         return "bottomright"
#     elif abs(x - rx) <= edge_size:
#         return "left"
#     elif abs(x - (rx + rw)) <= edge_size:
#         return "right"
#     elif abs(y - ry) <= edge_size:
#         return "top"
#     elif abs(y - (ry + rh)) <= edge_size:
#         return "bottom"
#     else:
#         return None

# def draw_roi(img, rect):
#     """ 绘制 ROI 和控制点 """
#     rx, ry, rw, rh = rect
#     cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
#     # 绘制四个角点
#     cv2.circle(img, (rx, ry), corner_size, (255, 0, 0), -1)
#     cv2.circle(img, (rx + rw, ry), corner_size, (255, 0, 0), -1)
#     cv2.circle(img, (rx, ry + rh), corner_size, (255, 0, 0), -1)
#     cv2.circle(img, (rx + rw, ry + rh), corner_size, (255, 0, 0), -1)

# def update_roi(x, y, edge, rect):
#     """ 根据边缘调整 ROI """
#     rx, ry, rw, rh = rect
#     if edge == "topleft":
#         rx, ry = x, y
#         rw, rh = rw + (ix - x), rh + (iy - y)
#     elif edge == "topright":
#         ry = y
#         rw, rh = x - rx, rh + (iy - y)
#     elif edge == "bottomleft":
#         rx = x
#         rw, rh = rw + (ix - x), y - ry
#     elif edge == "bottomright":
#         rw, rh = x - rx, y - ry
#     elif edge == "left":
#         rx = x
#         rw = rw + (ix - x)
#     elif edge == "right":
#         rw = x - rx
#     elif edge == "top":
#         ry = y
#         rh = rh + (iy - y)
#     elif edge == "bottom":
#         rh = y - ry
#     return rx, ry, rw, rh

# def mouse_callback(event, x, y, flags, param):
#     global ix, iy, drawing, resizing, roi_selected, selected_edge, rx, ry, rw, rh
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not roi_selected:
#             ix, iy = x, y
#             rx, ry, rw, rh = ix, iy, 0, 0
#             drawing = True
#         else:
#             selected_edge = find_selected_edge(x, y, (rx, ry, rw, rh))
#             if selected_edge:
#                 resizing = True
#                 ix, iy = x, y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             rx, ry = min(ix, x), min(iy, y)
#             rw, rh = abs(ix - x), abs(iy - y)
#         elif resizing:
#             rx, ry, rw, rh = update_roi(x, y, selected_edge, (rx, ry, rw, rh))

#     elif event == cv2.EVENT_LBUTTONUP:
#         if drawing:
#             drawing = False
#             roi_selected = True
#         elif resizing:
#             resizing = False

# # 创建窗口
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', mouse_callback)

# # 载入图像
# img = cv2.imread('../speckle_image/real/circle/nuclear_ring_ref.bmp')
# orig = img.copy()
# rx, ry, rw, rh = 0, 0, 0, 0

# while True:
#     img = orig.copy()
#     # 画出 ROI
#     draw_roi(img, (rx, ry, rw, rh))
#     cv2.imshow('image', img)
    
#     # 按 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# # 输出最终的 ROI
# print("Final ROI:", rx, ry, rw, rh)
