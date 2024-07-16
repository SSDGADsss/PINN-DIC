import cv2
import numpy as np

class ROISelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.roi = np.zeros(self.image.shape[:2], dtype=np.uint8)  # ROI mask
        self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.mode = 'rectangle'  # Can be 'rectangle' or 'circle'
        self.operation = 'add'  # Can be 'add' or 'subtract'
        self.start_point = None

    def draw_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.update_display(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
            if self.mode == 'rectangle':
                cv2.rectangle(self.new_roi, self.start_point, (x, y), 255, -1)
            elif self.mode == 'circle':
                radius = int(np.sqrt((x - self.start_point[0])**2 + (y - self.start_point[1])**2))
                cv2.circle(self.new_roi, self.start_point, radius, 255, -1)
            self.update_display(x, y, finalize=True)

    def update_display(self, x, y, finalize=False):
        temp_clone = self.image.copy()
        # Draw new ROI in blue
        new_roi_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.drawing or finalize:
            if self.mode == 'rectangle':
                cv2.rectangle(new_roi_mask, self.start_point, (x, y), 255, -1)
            elif self.mode == 'circle':
                radius = int(np.sqrt((x - self.start_point[0])**2 + (y - self.start_point[1])**2))
                cv2.circle(new_roi_mask, self.start_point, radius, 255, -1)
        temp_clone = cv2.addWeighted(temp_clone, 1, cv2.cvtColor(new_roi_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
        # Draw existing ROI in red
        temp_clone = cv2.addWeighted(temp_clone, 1, cv2.cvtColor(self.roi, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imshow("ROI Selector", temp_clone)

    def apply_operation(self):
        if self.operation == 'add':
            self.roi = cv2.bitwise_or(self.roi, self.new_roi)
        elif self.operation == 'subtract':
            self.roi = cv2.bitwise_and(self.roi, cv2.bitwise_not(self.new_roi))
        self.update_display(0, 0, finalize=True)

    def run(self):
        cv2.namedWindow("ROI Selector")
        cv2.setMouseCallback("ROI Selector", self.draw_roi)

        while True:
            cv2.imshow("ROI Selector", self.clone)
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
                if self.new_roi is not None:
                    self.apply_operation()
                self.new_roi = None
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.roi

if __name__ == "__main__":
    image_path = "../speckle_image/real/circle/nuclear_ring_ref.bmp"
    selector = ROISelector(image_path)
    final_roi = selector.run()
    cv2.imwrite('final_roi.png', final_roi)
