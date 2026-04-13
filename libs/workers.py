import time

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage
from skimage.feature import local_binary_pattern

import libs.dicom_helper as dicom
import libs.feature_extractor as extractor
import libs.visualization as viz


class UploadWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, QImage, str)
    error = pyqtSignal(str)

    def __init__(self, file_path, config):
        super().__init__()
        self.file_path = file_path
        self.config = config

    def run(self):
        try:
            # Preprocess the image for ML pipeline: - pixels_array was resized(256,256) for prediction,
            # display is retained at 1024x1024 for GUI display overlay heatmap.
            pixel_array, display = extractor.preprocess_cxr(self.file_path)

            # Emit fake progress while "processing"
            for i in range(1, 101):
                self.progress.emit(i)
                # small delay to simulate progress
                time.sleep(0.005)

            # Convert to QImage (grayscale or RGB) - using the display (1024x1024) version.
            if len(display.shape) == 2:
                h, w = display.shape
                qimage = QImage(display.data, w, h, w, QImage.Format_Grayscale8)
            elif len(display.shape) == 3 and display.shape[2] == 3:
                h, w, c = display.shape
                qimage = QImage(display.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                raise ValueError("Unsupported DICOM format")

            patient_id = dicom.read_dicom_data(self.file_path, "PatientID")

            # Perform LBP on the resized version
            lbp_histo = extractor.extract_feature_lbp(
                pixel_array, self.config.radius, self.config.method
            )

            # Perform LBP for display.
            lbp = extractor.extract_feature_lbp_only(
                display, self.config.radius, self.config.method
            )

            # Send display instead of resized because resized not needed in the app logic,
            # only lbp_histor is used. Meanwhile, lbp is used for display.
            self.finished.emit(display, lbp_histo, lbp, qimage, patient_id)

        except Exception as e:
            self.error.emit(str(e))


# Worker thread for asynchronous `analysis
class AnalyzeWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray)  # histogram, log_hist

    def __init__(self, pixel_array):
        super().__init__()
        self.pixel_array = pixel_array

    def run(self):
        # Parameters for LBP
        P = 8  # number of circularly symmetric neighbour set points
        R = 1  # radius
        method = "uniform"

        lbp = local_binary_pattern(self.pixel_array, P, R, method)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        log_hist = np.log1p(hist)  # log-scaled histogram

        # Simulate progress bar update for UI (optional)
        for i in range(1, 101):
            self.progress.emit(i)
            self.msleep(5)  # small delay to allow UI updates

        self.finished.emit(hist, log_hist)


class LBPOverlayWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(QImage)

    def __init__(self, sample_lbp, patient_lbp, pixel_array):
        super().__init__()
        self.pixel_array = pixel_array
        self.sample_lbp = sample_lbp
        self.patient_lbp = patient_lbp

    def run(self):
        import numpy as np
        from PyQt5.QtGui import QColor, QImage
        from skimage.feature import local_binary_pattern

        pixels_arr = viz.lbp_difference_map(
            self.sample_lbp, self.patient_lbp, self.pixel_array
        )
        overlay_img = self.numpy_to_qimage(pixels_arr)

        # self.finished.emit(self.startTimer, overlay_img)
        self.finished.emit(overlay_img)

    def numpy_to_qimage(self, array):
        # Ensure the array is uint8
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)  # Or other conversion

        height, width, channel = array.shape
        bytesPerLine = channel * width

        # Create QImage from buffer
        qimg = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # If using OpenCV (BGR), swap to RGB
        return qimg.rgbSwapped()

    def msleep(self, ms):
        """Sleep in milliseconds without blocking signals."""
        time.sleep(ms / 1000)


class DataLoadingWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # self.progress.emit(20)
        # Emit fake progress while "processing"
        for i in range(1, 78):
            self.progress.emit(i)
            time.sleep(0.005)  # small delay to simulate progress

        # Load X_train and y_train - initial version. This has the dataset size issue.
        # X_train, y_train = viz.load_dataset(
        #     self.config.dataset, "train", self.config.radius, self.config.method
        # )

        # The revised using h5 version to handle larger dataset size.
        X_train, y_train = viz.load_dataset_h5(self.config.train_dataset_path)

        self.progress.emit(100)
        self.finished.emit(X_train, y_train)
