import cv2
import numpy as np

class ROISelector:
    def __init__(self, image):
        self.original_image = image   # 一共三个画布：
        self.image = image.copy()     # 展示加减后的ROI
        self.clone = image.copy()     # 展示新画的ROI
        self.rois = []  # List to store multiple ROIs
        self.current_roi = None
        self.roi_type = 'rectangle'  # Default ROI type
        self.operation_mode = 'add'  # Default operation mode (add, subtract)
        self.drawing = False
        self.moving = False
        self.adjusting = False
        self.start_point = None
        self.end_point = None
        self.selected_roi_index = None

    def select_roi(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self._mouse_events)

        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.roi_type = 'rectangle'
            elif key == ord('c'):
                self.roi_type = 'circle'
            elif key == ord('a'):
                self.operation_mode = 'add'
            elif key == ord('s'):
                self.operation_mode = 'subtract'
            elif key == ord('m'):
                self.moving = True
            elif key == ord('e'):
                self.adjusting = True
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def _mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.adjusting:
                self._select_roi_for_adjustment((x, y))
            else:
                self.drawing = True
                self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.clone.copy()
                self._draw_roi((x, y))
            elif self.moving and self.selected_roi_index is not None:
                self._move_roi((x, y))
            elif self.adjusting and self.selected_roi_index is not None:
                self._adjust_roi((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self._finalize_roi((x, y))
            elif self.moving:
                self.moving = False
            elif self.adjusting:
                self.adjusting = False

    def _draw_roi(self, end_point):
        if self.roi_type == 'rectangle':
            cv2.rectangle(self.image, self.start_point, end_point, (0, 0, 255), 2)
        elif self.roi_type == 'circle':
            center = self.start_point
            radius = int(((end_point[0] - center[0]) ** 2 + (end_point[1] - center[1]) ** 2) ** 0.5)
            cv2.circle(self.image, center, radius, (0, 0, 255), 2)

    def _finalize_roi(self, end_point):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.roi_type == 'rectangle':
            cv2.rectangle(mask, self.start_point, end_point, 255, -1)
        elif self.roi_type == 'circle':
            center = self.start_point
            radius = int(((end_point[0] - center[0]) ** 2 + (end_point[1] - center[1]) ** 2) ** 0.5)
            cv2.circle(mask, center, radius, 255, -1)

        # Store ROI with start and end points for adjustment
        roi = {
            'mask': mask,
            'start_point': self.start_point,
            'end_point': end_point,
            'roi_type': self.roi_type
        }
        self.rois.append(roi)
        self._apply_rois()

    def _apply_rois(self):
        self.image = self.clone.copy()
        overlay = self.image.copy()

        for roi in self.rois:
            mask = roi['mask']
            overlay[mask > 0] = (0, 0, 255)  # Red color

        alpha = 0.5  # Transparency factor
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)

    def _select_roi_for_adjustment(self, point):
        for index, roi in enumerate(self.rois):
            if roi['mask'][point[1], point[0]] > 0:
                self.selected_roi_index = index
                break

    def _move_roi(self, point):
        if self.selected_roi_index is not None:
            roi = self.rois[self.selected_roi_index]
            dx = point[0] - roi['start_point'][0]
            dy = point[1] - roi['start_point'][1]

            if roi['roi_type'] == 'rectangle':
                roi['start_point'] = (roi['start_point'][0] + dx, roi['start_point'][1] + dy)
                roi['end_point'] = (roi['end_point'][0] + dx, roi['end_point'][1] + dy)
            elif roi['roi_type'] == 'circle':
                roi['start_point'] = (roi['start_point'][0] + dx, roi['start_point'][1] + dy)

            self.rois[self.selected_roi_index] = roi
            self._update_roi_mask(roi)
            self._apply_rois()

    def _adjust_roi(self, point):
        if self.selected_roi_index is not None:
            roi = self.rois[self.selected_roi_index]

            if roi['roi_type'] == 'rectangle':
                roi['end_point'] = point
            elif roi['roi_type'] == 'circle':
                # Adjust radius by changing end point
                roi['end_point'] = point

            self.rois[self.selected_roi_index] = roi
            self._update_roi_mask(roi)
            self._apply_rois()

    def _update_roi_mask(self, roi):
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if roi['roi_type'] == 'rectangle':
            cv2.rectangle(mask, roi['start_point'], roi['end_point'], 255, -1)
        elif roi['roi_type'] == 'circle':
            center = roi['start_point']
            radius = int(((roi['end_point'][0] - center[0]) ** 2 + (roi['end_point'][1] - center[1]) ** 2) ** 0.5)
            cv2.circle(mask, center, radius, 255, -1)

        roi['mask'] = mask

    def get_rois(self):
        return self.rois

# Usage example
if __name__ == "__main__":
    image = cv2.imread("../speckle_image/real/circle/nuclear_ring_ref.bmp")
    roi_selector = ROISelector(image)
    roi_selector.select_roi()

    for i, roi in enumerate(roi_selector.get_rois()):
        cv2.imshow(f"ROI {i+1}", roi)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
