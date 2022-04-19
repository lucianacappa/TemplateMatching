import cv2
import numpy


class PlotWindow:
    INITIAL_SCALE = 1.0
    MAX_SCALE = 5.0
    MIN_SCALE = 0.2
    SCALE_STEP = 0.1
    WINDOW_NAME = 'plot_window'

    def __init__(self, dimensions: tuple[int, int] = (600, 800), center: tuple[int, int] = None,
                 scale: float = INITIAL_SCALE):
        self._dragging = None
        self._full_plot_image = None
        self._initial_center = center
        self._initial_scale = scale
        self._title = None
        self.center = center
        self.scale = scale
        cv2.namedWindow(PlotWindow.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PlotWindow.WINDOW_NAME, dimensions[1], dimensions[0])
        cv2.setMouseCallback(PlotWindow.WINDOW_NAME, self._on_mouse)

    def _crop_to_win_size(self):
        ih, iw = self._scaled_plot_image.shape[:2]
        cx, cy = self.center
        wh, ww = self.get_window_dimensions()
        half_wh, half_ww = (wh // 2, ww // 2)
        top = int(max(cy - half_wh, 0))
        bottom = int(min(cy - half_wh + wh, ih))
        left = int(max(cx - half_ww, 0))
        right = int(min(cx - half_ww + ww, iw))
        self._cropped_plot_image = numpy.zeros((wh, ww, 3), numpy.uint8)
        self._cropped_plot_image[0:bottom - top, 0:right - left] = \
            self._scaled_plot_image[top:bottom, left:right]

    def _fix_center(self):
        ih, iw = self._scaled_plot_image.shape[:2]
        cx, cy = (ih // 2, iw // 2) if self.center is None else self.center
        wh, ww = self.get_window_dimensions()
        if cx < ww // 2:
            cx = ww // 2
        elif cx > iw - ww // 2:
            cx = iw - ww // 2
        if cy < wh // 2:
            cy = wh // 2
        elif cy > ih - wh // 2:
            cy = ih - wh // 2
        self.center = (cx, cy)

    # noinspection PyMethodMayBeStatic
    def get_window_dimensions(self):
        return cv2.getWindowImageRect(PlotWindow.WINDOW_NAME)[2:4][::-1]

    def _on_mouse(self, event: int, x: int, y: int, flag: int, userdata):
        if flag & cv2.EVENT_FLAG_LBUTTON:
            if self._dragging is not None:
                self.move_center((self._dragging[0] - x, self._dragging[1] - y))
            self._dragging = (x, y)
        elif self._dragging is not None and (not flag & cv2.EVENT_FLAG_LBUTTON):
            self._dragging = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flag > 0:
                self.scale_up((x, y))
            else:
                self.scale_down((x, y))
            self.set_title(self._title)

    def _scale_plot(self):
        self._scaled_plot_image = cv2.resize(self._full_plot_image, (0, 0), fx=self.scale, fy=self.scale)
        self._fix_center()
        self._show()

    def _show(self):
        self._crop_to_win_size()
        cv2.imshow(PlotWindow.WINDOW_NAME, self._cropped_plot_image)

    # noinspection PyMethodMayBeStatic
    def destroy(self):
        cv2.destroyWindow(PlotWindow.WINDOW_NAME)

    def move_center(self, delta: tuple[int, int]):
        self.center = (self.center[0] + delta[0], self.center[1] + delta[1])
        self._fix_center()
        self._show()

    def redraw(self):
        self._scale_plot()

    def reset_center(self):
        self.center = self._initial_center
        self._scale_plot()

    def reset_window_params(self):
        self.center = self._initial_center
        self.scale = self._initial_scale
        self._scale_plot()

    def scale_down(self, at: tuple[int, int]):
        if self.scale > PlotWindow.MIN_SCALE:
            self.scale -= PlotWindow.SCALE_STEP
        self._scale_plot()

    def scale_up(self, at: tuple[int, int]):
        if self.scale < PlotWindow.MAX_SCALE:
            self.scale += PlotWindow.SCALE_STEP
        self._scale_plot()

    def set_plot_image(self, plot_image, title: str):
        self._full_plot_image = plot_image
        self._scale_plot()
        self.set_title(title)

    def set_title(self, title):
        self._title = title
        cv2.setWindowTitle(PlotWindow.WINDOW_NAME, f'{title} (scale = {self.scale * 100.0:.1f}%)')
