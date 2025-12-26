"""
Drill-down viewer for detailed anomaly analysis

Shows 4-panel layout:
- Time waveform
- FFT spectrum
- Envelope spectrum (demodulated)
- NeuraPeak spectrum (stress wave)

All synchronized to the same time/data point
"""

import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from scipy import signal


class DrillDownViewer(QtWidgets.QWidget):
    """
    Multi-panel viewer for detailed signal analysis

    Layout:
        ┌──────────────┬──────────────┐
        │  Waveform    │  FFT         │
        ├──────────────┼──────────────┤
        │  Envelope    │  NeuraPeak   │
        └──────────────┴──────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.bearing_freqs = None

    def _init_ui(self):
        """Initialize 2x2 grid layout"""
        layout = QtWidgets.QVBoxLayout()

        # Title and info panel
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.info_label)

        # Create 2x2 grid for plots
        grid_layout = QtWidgets.QGridLayout()

        # Create 4 plot widgets
        self.waveform_plot = pg.PlotWidget(title="Time Waveform")
        self.waveform_plot.setLabel('left', 'Amplitude', units='g')
        self.waveform_plot.setLabel('bottom', 'Time', units='s')
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)

        self.fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.fft_plot.setLabel('left', 'Amplitude', units='g')
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.addLegend()

        self.envelope_plot = pg.PlotWidget(title="Envelope Spectrum (Demodulated)")
        self.envelope_plot.setLabel('left', 'Amplitude')
        self.envelope_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.envelope_plot.showGrid(x=True, y=True, alpha=0.3)

        self.neurapeak_plot = pg.PlotWidget(title="NeuraPeak™ Spectrum (Stress Wave)")
        self.neurapeak_plot.setLabel('left', 'Amplitude')
        self.neurapeak_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.neurapeak_plot.showGrid(x=True, y=True, alpha=0.3)

        # Add to grid
        grid_layout.addWidget(self.waveform_plot, 0, 0)
        grid_layout.addWidget(self.fft_plot, 0, 1)
        grid_layout.addWidget(self.envelope_plot, 1, 0)
        grid_layout.addWidget(self.neurapeak_plot, 1, 1)

        layout.addLayout(grid_layout)

        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.export_all_btn = QtWidgets.QPushButton('📊 Export All Charts')
        self.export_all_btn.clicked.connect(self._export_all)
        button_layout.addWidget(self.export_all_btn)

        self.export_data_btn = QtWidgets.QPushButton('💾 Download Raw Data')
        self.export_data_btn.clicked.connect(self._export_raw_data)
        button_layout.addWidget(self.export_data_btn)

        button_layout.addStretch()

        self.close_btn = QtWidgets.QPushButton('✖ Close')
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setWindowTitle('MachineIQ Anomaly Drill-Down')
        self.resize(1600, 900)

        # Store data for export
        self.current_data = {}

    def load_anomaly_data(self,
                         channel_name: str,
                         timestamp: float,
                         waveform: np.ndarray,
                         sample_rate: float,
                         bearing_freqs: Optional[Dict[str, float]] = None):
        """
        Load and display data for an anomaly

        Args:
            channel_name: Name of the channel
            timestamp: Unix timestamp of anomaly
            waveform: Raw time-domain waveform
            sample_rate: Sampling rate in Hz
            bearing_freqs: Dict with 'BPFO', 'BPFI', 'BSF', 'FTF' keys
        """
        self.bearing_freqs = bearing_freqs or {}
        self.current_data = {
            'channel': channel_name,
            'timestamp': timestamp,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'bearing_freqs': bearing_freqs
        }

        # Update info label
        dt = datetime.fromtimestamp(timestamp)
        info_text = f"<b>Channel:</b> {channel_name} | "
        info_text += f"<b>Time:</b> {dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        info_text += f"<b>Sample Rate:</b> {sample_rate} Hz | "
        info_text += f"<b>Points:</b> {len(waveform)}"
        self.info_label.setText(info_text)

        # Clear all plots
        self.waveform_plot.clear()
        self.fft_plot.clear()
        self.envelope_plot.clear()
        self.neurapeak_plot.clear()

        # 1. Time waveform
        time = np.arange(len(waveform)) / sample_rate
        self.waveform_plot.plot(time, waveform, pen='b')

        # Add RMS line
        rms_value = np.sqrt(np.mean(waveform**2))
        rms_line = pg.InfiniteLine(
            pos=rms_value,
            angle=0,
            pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine),
            label=f'RMS: {rms_value:.3f}g'
        )
        self.waveform_plot.addItem(rms_line)

        # 2. FFT Spectrum
        fft_result = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(len(waveform), d=1/sample_rate)
        fft_magnitude = np.abs(fft_result)

        self.fft_plot.plot(freqs, fft_magnitude, pen='r', name='FFT')

        # Add bearing frequency markers if provided
        if bearing_freqs:
            self._add_bearing_markers(self.fft_plot, bearing_freqs, "FFT")

        # 3. Envelope Spectrum (Demodulated)
        envelope_signal, envelope_freqs, envelope_spectrum = self._compute_envelope_spectrum(
            waveform, sample_rate
        )

        if envelope_spectrum is not None:
            self.envelope_plot.plot(envelope_freqs, envelope_spectrum, pen='g')
            if bearing_freqs:
                self._add_bearing_markers(self.envelope_plot, bearing_freqs, "Envelope")

        # 4. NeuraPeak Spectrum (Stress Wave)
        neurapeak_freqs, neurapeak_spectrum = self._compute_neurapeak_spectrum(
            waveform, sample_rate
        )

        if neurapeak_spectrum is not None:
            self.neurapeak_plot.plot(neurapeak_freqs, neurapeak_spectrum, pen='orange')
            if bearing_freqs:
                self._add_bearing_markers(self.neurapeak_plot, bearing_freqs, "NeuraPeak")

    def _compute_envelope_spectrum(self, waveform: np.ndarray,
                                   sample_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute envelope spectrum using demodulation

        Returns:
            (envelope_signal, frequencies, spectrum)
        """
        try:
            # Bandpass filter (typical bearing defect frequency range)
            nyquist = sample_rate / 2
            low_cutoff = min(500, nyquist * 0.1)
            high_cutoff = min(5000, nyquist * 0.8)

            sos = signal.butter(4, [low_cutoff, high_cutoff],
                               btype='band', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, waveform)

            # Hilbert transform to get envelope
            analytic_signal = signal.hilbert(filtered)
            envelope_signal = np.abs(analytic_signal)

            # FFT of envelope
            envelope_fft = np.fft.rfft(envelope_signal)
            envelope_freqs = np.fft.rfftfreq(len(envelope_signal), d=1/sample_rate)
            envelope_spectrum = np.abs(envelope_fft)

            return envelope_signal, envelope_freqs, envelope_spectrum

        except Exception as e:
            print(f"Envelope computation failed: {e}")
            return None, None, None

    def _compute_neurapeak_spectrum(self, waveform: np.ndarray,
                                    sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute NeuraPeak (stress wave) spectrum

        This is a simplified version of the patented PeakVue algorithm:
        1. High-pass filter (remove low-frequency content)
        2. Rectify (take absolute value or square)
        3. Low-pass filter (extract envelope)
        4. FFT of result
        """
        try:
            nyquist = sample_rate / 2

            # High-pass filter to isolate stress waves
            hp_cutoff = min(5000, nyquist * 0.4)
            sos_hp = signal.butter(4, hp_cutoff, btype='high', fs=sample_rate, output='sos')
            filtered_hp = signal.sosfilt(sos_hp, waveform)

            # Rectify (square for energy)
            rectified = filtered_hp ** 2

            # Low-pass filter to extract envelope
            lp_cutoff = min(2000, nyquist * 0.2)
            sos_lp = signal.butter(4, lp_cutoff, btype='low', fs=sample_rate, output='sos')
            envelope = signal.sosfilt(sos_lp, rectified)

            # Decimate if needed for performance
            decim_factor = 4
            if len(envelope) > 10000:
                envelope = signal.decimate(envelope, decim_factor)
                effective_sr = sample_rate / decim_factor
            else:
                effective_sr = sample_rate

            # FFT
            neurapeak_fft = np.fft.rfft(envelope)
            neurapeak_freqs = np.fft.rfftfreq(len(envelope), d=1/effective_sr)
            neurapeak_spectrum = np.abs(neurapeak_fft)

            return neurapeak_freqs, neurapeak_spectrum

        except Exception as e:
            print(f"NeuraPeak computation failed: {e}")
            return None, None

    def _add_bearing_markers(self, plot_widget: pg.PlotWidget,
                            bearing_freqs: Dict[str, float], label_prefix: str = ""):
        """Add vertical lines for bearing fault frequencies"""
        colors = {
            'BPFO': (255, 255, 0),  # Yellow
            'BPFI': (0, 255, 0),    # Green
            'BSF': (0, 255, 255),   # Cyan
            'FTF': (255, 0, 255)    # Magenta
        }

        for name, freq in bearing_freqs.items():
            if freq is not None and freq > 0:
                line = pg.InfiniteLine(
                    pos=freq,
                    angle=90,
                    pen=pg.mkPen(colors.get(name, (255, 255, 255)),
                                width=2, style=QtCore.Qt.PenStyle.DashLine),
                    label=f"{name}: {freq:.1f}Hz"
                )
                plot_widget.addItem(line)

                # Add harmonics (2x, 3x)
                for harmonic in [2, 3]:
                    harmonic_freq = freq * harmonic
                    harmonic_line = pg.InfiniteLine(
                        pos=harmonic_freq,
                        angle=90,
                        pen=pg.mkPen(colors.get(name, (255, 255, 255)),
                                    width=1, style=QtCore.Qt.PenStyle.DotLine),
                        label=f"{name}×{harmonic}"
                    )
                    plot_widget.addItem(harmonic_line)

    def _export_all(self):
        """Export all four plots as PNG"""
        from pyqtgraph.exporters import ImageExporter

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Export Directory'
        )

        if directory:
            plots = [
                (self.waveform_plot, 'waveform'),
                (self.fft_plot, 'fft'),
                (self.envelope_plot, 'envelope'),
                (self.neurapeak_plot, 'neurapeak')
            ]

            timestamp_str = datetime.fromtimestamp(
                self.current_data['timestamp']
            ).strftime('%Y%m%d_%H%M%S')

            for plot, name in plots:
                filename = f"{directory}/{self.current_data['channel']}_{name}_{timestamp_str}.png"
                exporter = ImageExporter(plot.plotItem)
                exporter.parameters()['width'] = 1920
                exporter.export(filename)

            QtWidgets.QMessageBox.information(
                self,
                'Export Success',
                f'All charts exported to:\n{directory}'
            )

    def _export_raw_data(self):
        """Export raw waveform data as CSV"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Raw Data', '', 'CSV Files (*.csv)'
        )

        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'

            waveform = self.current_data['waveform']
            sample_rate = self.current_data['sample_rate']
            time = np.arange(len(waveform)) / sample_rate

            # Write CSV
            with open(filename, 'w') as f:
                f.write('Time (s),Amplitude (g)\n')
                for t, amp in zip(time, waveform):
                    f.write(f'{t:.6f},{amp:.6f}\n')

            QtWidgets.QMessageBox.information(
                self,
                'Export Success',
                f'Raw data exported to:\n{filename}'
            )


# Demo function
def demo_drill_down():
    """Demonstrate the drill-down viewer"""
    import sys

    app = QtWidgets.QApplication(sys.argv)

    viewer = DrillDownViewer()

    # Simulated bearing fault data
    sample_rate = 25600  # Hz
    duration = 1.0  # second
    num_samples = int(sample_rate * duration)
    time = np.linspace(0, duration, num_samples)

    # Simulate BPFO fault at 120 Hz (e.g., 1800 RPM / 60 * 4 defects)
    shaft_speed = 30  # Hz (1800 RPM)
    bpfo_freq = 120
    bpfi_freq = 180
    bsf_freq = 45
    ftf_freq = 12

    # Create synthetic bearing fault signal
    waveform = (
        1.0 * np.sin(2 * np.pi * shaft_speed * time) +  # 1x shaft speed
        0.5 * np.sin(2 * np.pi * shaft_speed * 2 * time) +  # 2x shaft speed
        3.0 * np.sin(2 * np.pi * bpfo_freq * time) +  # BPFO fault
        0.8 * np.sin(2 * np.pi * bpfo_freq * 2 * time) +  # BPFO 2x harmonic
        0.5 * np.random.randn(num_samples)  # Noise
    )

    # Add some impulses to simulate bearing impacts
    for i in range(0, num_samples, int(sample_rate / bpfo_freq)):
        if i < num_samples:
            waveform[i:i+10] += 5.0 * np.exp(-np.arange(10) / 3)

    bearing_freqs = {
        'BPFO': bpfo_freq,
        'BPFI': bpfi_freq,
        'BSF': bsf_freq,
        'FTF': ftf_freq
    }

    viewer.load_anomaly_data(
        channel_name="DE Radial Accelerometer",
        timestamp=datetime.now().timestamp(),
        waveform=waveform,
        sample_rate=sample_rate,
        bearing_freqs=bearing_freqs
    )

    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    demo_drill_down()
