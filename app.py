import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import (
    QCoreApplication,
    QFile,
    QLockFile,
    QSharedMemory,
    QSystemSemaphore,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import libs.installer_lib as installer
from config import app_config
from libs import dicom_helper as dicom
from libs import feature_extractor as extractor
from libs import visualization as viz
from libs.workers import (
    AnalyzeWorker,
    DataLoadingWorker,
    LBPOverlayWorker,
    UploadWorker,
)

# class SingleInstanceApp(QApplication):
#     def __init__(self, argv):
#         super().__init__(argv)

#         # Define a system-wide semaphore for shared memory
#         self.semaphore = QSystemSemaphore("SingleAppSemaphore", QSystemSemaphore.Create)
#         self.semaphore.acquire()  # Try to acquire the semaphore lock

#         # Shared memory block for instance checking
#         self.shared_memory = QSharedMemory("SingleAppSharedMemory")

#         if self.shared_memory.attach():
#             # Another instance is running
#             print("Another instance is already running.")
#             sys.exit(0)  # Exit the current instance

#         # Attach the shared memory to block others
#         self.shared_memory.create(1)

#     def __del__(self):
#         self.shared_memory.detach()  # Detach the shared memory when the app exits
#         self.semaphore.release()  # Release the semaphore


class SingleInstanceApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

        # Path to a lock file (this file will be used to check if an instance is already running)
        self.lock_file = QLockFile(installer.resource_path("myapp.lock"))

        # Try to lock the file, if it fails, that means an instance is already running
        if not self.lock_file.tryLock():
            print("Another instance is already running.")
            sys.exit(0)  # Exit the current instance

    def __del__(self):
        # Release the lock on the file when the app exits
        self.lock_file.unlock()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediViz - prototype")
        self.setFixedSize(2000, 1200)
        self.setup_ui()
        self.init_app()

    def setup_ui(self):
        # Main vertical layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Dashboard Panel (90%)
        dashboard_panel = QFrame()
        dashboard_panel.setFrameShape(QFrame.StyledPanel)
        dashboard_layout = QHBoxLayout(dashboard_panel)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(5)

        # Viewer Panel (60%)
        self.viewer_panel = QFrame()
        self.viewer_panel.setFrameShape(QFrame.StyledPanel)
        self.viewer_panel.resizeEvent = self.update_lbp_position

        # Use a vertical layout
        self.viewer_layout = QVBoxLayout(self.viewer_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_layout.setSpacing(0)

        # Spacer above image to center vertically
        self.top_spacer = QVBoxLayout()
        self.top_spacer.addStretch(1)
        self.viewer_layout.addLayout(self.top_spacer)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(QPixmap("resources/no-image.png"))
        self.viewer_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Spacer below image to center vertically
        self.bottom_spacer = QVBoxLayout()
        self.bottom_spacer.addStretch(1)
        self.viewer_layout.addLayout(self.bottom_spacer)

        # LBP Overlay Radio Button at bottom-right (absolute alignment)
        self.lbp_radio = QRadioButton("LBP overlay", self.viewer_panel)
        self.lbp_radio.move(
            self.viewer_panel.width() - 110, self.viewer_panel.height() - 30
        )
        self.lbp_radio.setEnabled(False)
        self.lbp_radio.show()

        # --- Info Panel (40%) ---
        self.info_panel = QFrame()
        self.info_panel.setFrameShape(QFrame.StyledPanel)
        self.info_layout = QVBoxLayout(self.info_panel)

        # Prediction section (10% of height)
        self.prediction_div = QFrame()
        self.prediction_layout = QHBoxLayout(self.prediction_div)
        self.prediction_div.setStyleSheet(
            f"""
            border-radius: 8px;
            padding: 5px;
        """
        )
        self.prediction_label = QLabel("Pneumonia ?")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet(
            """
            font-size: 18px;  
            font-weight: bold;
        """
        )
        self.prediction_result_label = QLabel("NA")
        self.prediction_result_label.setAlignment(Qt.AlignCenter)
        self.prediction_result_label.setStyleSheet(
            """
            font-size: 18px;  
            font-weight: bold;
        """
        )
        self.prediction_layout.addWidget(self.prediction_label, 30)
        self.prediction_layout.addWidget(self.prediction_result_label, 70)
        self.info_layout.addWidget(self.prediction_div, 10)

        # Scatter plot section (45% of height)
        self.scatter_plot_div = QFrame()
        self.scatter_plot_layout = QVBoxLayout(self.scatter_plot_div)

        # Add a spacer between scatter and histogram
        self.scatter_spacer = QSpacerItem(
            20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding
        )
        self.info_layout.addItem(self.scatter_spacer)
        self.info_layout.addWidget(self.scatter_plot_div, 45)

        # Histogram section (45% of height)
        self.histo_div = QFrame()
        self.histo_layout = QVBoxLayout(self.histo_div)

        # Add the histogram section
        self.info_layout.addWidget(self.histo_div, 45)

        # Add panels to dashboard layout
        dashboard_layout.addWidget(self.viewer_panel, 60)
        dashboard_layout.addWidget(self.info_panel, 40)

        # Control Panel (10%)
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(5)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)

        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setEnabled(True)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setEnabled(False)

        for btn in [self.upload_btn, self.analyze_btn, self.clear_btn]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            buttons_layout.addWidget(btn)

        control_layout.addLayout(buttons_layout)

        # Add panels to main layout
        main_layout.addWidget(dashboard_panel, 90)
        main_layout.addWidget(control_panel, 10)

        # Connect buttons to handlers
        self.upload_btn.clicked.connect(self.upload_action)
        self.analyze_btn.clicked.connect(self.analyze_action)
        self.clear_btn.clicked.connect(self.clear_action)
        self.lbp_radio.toggled.connect(self.toggle_lbp_overlay)

    def init_app(self):
        # Load app config.
        self.config = app_config

        # Load sample normal CXR - the original size (1024x1024).
        _, self.sample_normal_cxr_img = extractor.preprocess_cxr(
            self.config.sample_normal_cxr
        )

        # LBP of sample normal CXR.
        self.sample_normal_cxr_lbp = extractor.extract_feature_lbp_only(
            self.sample_normal_cxr_img, self.config.radius, self.config.method
        )

        # Load pickle file into model.
        self.model = viz.load_model(self.config.model_path)

        self.download_thread()

    def download_thread(self):
        # Disable upload button while initializing.
        self.upload_btn.setEnabled(False)

        self.ld_thread = QThread()
        self.ld_worker = DataLoadingWorker(self.config)
        self.ld_worker.moveToThread(self.ld_thread)

        # Connect signals
        self.ld_thread.started.connect(self.ld_worker.run)
        self.ld_worker.progress.connect(self.progress_bar.setValue)
        self.ld_worker.finished.connect(self.set_data)
        self.ld_thread.finished.connect(self.ld_worker.deleteLater)
        self.ld_thread.finished.connect(self.ld_thread.deleteLater)

        # Start the thread
        self.ld_thread.start()

    def set_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.upload_btn.setEnabled(True)

        # Reset progress bar
        self.progress_bar.setValue(0)

    def update_lbp_position(self, event):
        """Keep LBP radio button at bottom-right corner."""
        x = self.viewer_panel.width() - self.lbp_radio.width() - 10
        y = self.viewer_panel.height() - self.lbp_radio.height() - 10
        self.lbp_radio.move(x, y)

    def upload_action(self):
        """Prompt file dialog and start UploadWorker thread."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DICOM Image", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if not file_path:
            return
        self.img_path = file_path

        # Disable buttons during upload
        self.upload_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Create thread and worker
        self.upload_thread = QThread()
        self.upload_worker = UploadWorker(self.img_path, self.config)
        self.upload_worker.moveToThread(self.upload_thread)

        # Connect signals
        self.upload_thread.started.connect(self.upload_worker.run)
        self.upload_worker.progress.connect(self.progress_bar.setValue)
        self.upload_worker.finished.connect(self.display_uploaded_image)
        self.upload_worker.finished.connect(self.upload_thread.quit)
        self.upload_worker.finished.connect(self.upload_worker.deleteLater)
        self.upload_thread.finished.connect(self.upload_thread.deleteLater)
        self.upload_worker.error.connect(self.upload_error)

        # Start thread
        self.upload_thread.start()

    def display_uploaded_image(self, pixel_array, lbp_histo, lbp, qimage, patient_id):
        # Store pixel array and QImage for later analysis
        self.current_pixel_array = pixel_array
        self.current_qimage = qimage
        self.lbp_histo = lbp_histo
        self.lbp = lbp
        self.patient_id = patient_id

        # Display image
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Display Patient ID at top
        if hasattr(self, "patient_label"):
            self.patient_label.setText(f"Patient ID: {patient_id}")
        else:
            self.patient_label = QLabel(
                f"Patient ID: {patient_id} (Original resolution)"
            )
            self.patient_label.setAlignment(Qt.AlignCenter)
            font = self.patient_label.font()
            font.setPointSize(16)
            font.setBold(True)
            self.patient_label.setFont(font)
            self.patient_label.setStyleSheet("color: lightgreen;")
            self.viewer_layout.insertWidget(0, self.patient_label)

        # Enable buttons
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.lbp_radio.setEnabled(True)

        # Reset progress bar
        self.progress_bar.setValue(0)

    def upload_error(self, msg):
        print("Upload error:", msg)
        self.progress_bar.setValue(0)
        self.upload_btn.setEnabled(True)

    def analyze_action(self):
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        # Disable clear button while analysis
        self.clear_btn.setEnabled(False)

        self.plot_scatter(self.scatter_plot_layout)
        self.plot_histo(self.histo_layout, self.config.method)

        self.predict_outcome()
        self.update_prediction()

        # Enable button post analysis
        self.clear_btn.setEnabled(True)

    def update_prediction(self):
        # Set the text
        self.prediction_result_label.setText(self.result_text)

        # Set the background color of the div
        if self.pneumonia:
            bg_color = "#ff6b6b"
        else:
            bg_color = "#90ee90"

        self.prediction_div.setStyleSheet(
            f"""
            background-color: {bg_color};
            border-radius: 8px;
            padding: 5px;
        """
        )

    def predict_outcome(self):
        if self.config.lda:
            x_new_lda = extractor.perform_single_lda(
                self.X_train, self.y_train, self.lbp_histo.reshape(1, -1), 1
            )
            result = self.model.predict(x_new_lda)
            proba = self.model.predict_proba(x_new_lda)
            print(result)
            print(proba)
        else:
            result = self.model.predict(self.lbp_histo.reshape(1, -1))
            proba = self.model.predict_proba(self.lbp_histo.reshape(1, -1))
            print(result)
            print(proba)

        # Format pneumonia
        self.pneumonia = bool(result.item())

        # Format probability
        # Convert probabilities to percentages and round
        normal_pct = round(proba[0, 0] * 100)
        pneumonia_pct = round(proba[0, 1] * 100)

        # Format as string
        self.result_text = f"{"Yes" if result else "No"}: Probabilty [Normal={normal_pct}%, Pneumonia={pneumonia_pct}%]"

    def plot_scatter(self, qvbox):
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Create figure + axis
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Scatter: Normal
        ax.scatter(
            X_pca[self.y_train == 0, 0],
            X_pca[self.y_train == 0, 1],
            color="blue",
            label="Normal",
            alpha=0.7,
        )

        # Scatter: Pneumonia
        ax.scatter(
            X_pca[self.y_train == 1, 0],
            X_pca[self.y_train == 1, 1],
            color="red",
            label="Pneumonia",
            alpha=0.7,
        )

        if self.patient_id is not None and self.patient_id != "":
            pre_normal, _ = extractor.preprocess_cxr(self.img_path)
            X_new = extractor.extract_feature_lbp(
                pre_normal, self.config.radius, self.config.method
            )
            X_new = X_new.reshape(1, -1)

            # Transform with SAME scaler & PCA
            X_new_scaled = scaler.transform(X_new)
            X_new_pca = pca.transform(X_new_scaled)

            ax.scatter(
                X_new_pca[0, 0],
                X_new_pca[0, 1],
                color="green",
                s=200,
                marker="*",
                label="Patient",
                edgecolors="black",
            )

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("PCA (with training data): Normal vs Pneumonia with Patient")

        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        qvbox.addWidget(canvas)

    def plot_histo(self, qvbox, method):
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        if method == "uniform":
            # skimage uniform patterns formula
            # P=neighbours
            P = 8
            n_bins = int(P * (P - 1) + 3)
            # n_bins = 20
            bins = np.arange(0, n_bins + 1)
            max_val = n_bins - 1
        else:
            bins = np.arange(0, 257)
            max_val = 255

        print(f"uniform:{method}")
        print(f"bin:{bins}")

        # Plot normal sample histograms
        ax.hist(
            self.sample_normal_cxr_lbp.ravel(),
            bins=bins,
            density=True,
            color="royalblue",
            label="Normal",
            histtype="step",
            linewidth=2,
            log=True,
        )
        # Plot patient histogram
        ax.hist(
            self.lbp.ravel(),
            bins=bins,
            density=True,
            color="crimson",
            label="Uploaded",
            histtype="step",
            linewidth=1.5,
            log=True,
        )
        title = f"Comparison of LBP {self.config.method} Texture Signatures"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"LBP Pattern Value (0-{max_val})", fontsize=12)
        ax.set_ylabel("Log Probability Density", fontsize=12)
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, which="both", linestyle="-", alpha=0.1)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        qvbox.addWidget(canvas)

    def toggle_lbp_overlay(self, checked):
        if not checked:
            # Restore original CXR safely
            if self.current_qimage is not None:
                self.image_label.setPixmap(QPixmap.fromImage(self.current_qimage))
            else:
                self.image_label.setPixmap(
                    QPixmap("resources/no-image.png").scaled(
                        600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            return

        # Disable radio button while computing
        self.lbp_radio.setEnabled(False)

        # Keep thread as instance attribute to prevent garbage collection
        self.lbp_thread = QThread()
        self.lbp_worker = LBPOverlayWorker(
            self.sample_normal_cxr_lbp, self.lbp, self.current_pixel_array
        )
        self.lbp_worker.moveToThread(self.lbp_thread)

        # Connect signals
        self.lbp_thread.started.connect(self.lbp_worker.run)
        self.lbp_worker.progress.connect(self.progress_bar.setValue)
        self.lbp_worker.finished.connect(self.apply_lbp_overlay)
        self.lbp_worker.finished.connect(self.lbp_thread.quit)  # quit thread first
        self.lbp_thread.finished.connect(
            self.lbp_worker.deleteLater
        )  # delete worker after quit
        self.lbp_thread.finished.connect(
            self.lbp_thread.deleteLater
        )  # delete thread after quit

        # Start the thread
        self.lbp_thread.start()

    def apply_lbp_overlay(self, overlay_img):
        """Overlay the LBP image on the original CXR."""
        from PyQt5.QtGui import QPainter

        # Convert stored original QImage to ARGB
        base_img = self.current_qimage.convertToFormat(QImage.Format_ARGB32)
        painter = QPainter(base_img)
        painter.drawImage(0, 0, overlay_img)
        painter.end()

        # Display the combined image
        self.image_label.setPixmap(QPixmap.fromImage(base_img))
        self.lbp_radio.setEnabled(True)  # re-enable radio button
        self.progress_bar.setValue(0)

    def clear_action(self):
        """Clear the UI, reset image, info panel, and return to the initial state."""
        # Reset the image label to the initial "no image" placeholder
        self.image_label.setPixmap(
            QPixmap("resources/no-image.png").scaled(
                600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        # Reset the progress bar to 0
        self.progress_bar.setValue(0)

        # Disable Analyze and Clear buttons
        self.analyze_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Reset Patient ID label if it exists
        if hasattr(self, "patient_label"):
            self.patient_label.setText("")

        # Clear data.
        self.current_pixel_array = None
        self.current_qimage = None
        self.lbp = None
        self.img_path = None
        self.pneumonia = False
        self.result_text = None

        # Reset the LBP radio button to disabled
        self.lbp_radio.setChecked(False)
        self.lbp_radio.setEnabled(False)

        # Prediction labels
        self.prediction_result_label.setText("None")

        # Remove background color (reset to default)
        self.prediction_div.setStyleSheet("")

        # Re-enable the Upload button
        self.upload_btn.setEnabled(True)

        for i in reversed(range(self.histo_layout.count())):
            widget = self.histo_layout.itemAt(i).widget()
            if widget is not None:
                # Delete the widget (clear histogram)
                widget.deleteLater()

        for i in reversed(range(self.scatter_plot_layout.count())):
            widget = self.scatter_plot_layout.itemAt(i).widget()
            if widget is not None:
                # Delete the widget (clear histogram)
                widget.deleteLater()

        # Optionally reset the layout if you want a completely clean state
        self.histo_layout.update()
        self.scatter_plot_layout.update()


if __name__ == "__main__":
    # To ensure only one instance of the application runs, we use SingleInstanceApp instead of QApplication.
    # app = QApplication(sys.argv)
    app = SingleInstanceApp(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
