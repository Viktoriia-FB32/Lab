import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Button, ColumnDataSource, Select

class HarmonicVisualizer:
    def __init__(self):
        # Параметри сигналу
        self.time = np.linspace(0, 10, 1000)
        self.amplitude = 1.0
        self.frequency = 1.0
        self.phase = 0.0
        self.noise_mean = 0.0
        self.noise_cov = 0.1
        
        # Параметри фільтру
        self.filter_types = ["Низькочастотний", "Високочастотний", "Смуговий"]
        self.filter_type = "Низькочастотний"
        self.filter_order = 4
        self.cutoff_freq = 2.0
        
        # Генерація даних
        self.generate_data()
        
        # Джерела даних для графіків
        self.source_time = ColumnDataSource(data={
            'time': self.time,
            'pure': self.pure_signal,
            'noisy': self.noisy_signal,
            'filtered': self.filtered_signal
        })
        
        self.source_freq = ColumnDataSource(data={
            'freq': self.freq_axis,
            'fft_pure': self.fft_pure,
            'fft_noisy': self.fft_noisy,
            'fft_filtered': self.fft_filtered
        })
        
        # Створення графіків
        self.create_plots()
        self.create_controls()
        self.setup_callbacks()
        
    def generate_data(self):
        """Генерує сигнал, шум і застосовує фільтр."""
        self.pure_signal = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
        self.noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_cov), len(self.time))
        self.noisy_signal = self.pure_signal + self.noise
        self.filtered_signal = self.apply_custom_filter(self.noisy_signal)
        self.calculate_fft()
    
    def calculate_fft(self):
        """Обчислює FFT для всіх сигналів."""
        n = len(self.time)
        dt = self.time[1] - self.time[0]
        
        self.freq_axis = np.fft.fftfreq(n, dt)[:n//2]
        self.fft_pure = np.abs(np.fft.fft(self.pure_signal)[:n//2])
        self.fft_noisy = np.abs(np.fft.fft(self.noisy_signal)[:n//2])
        self.fft_filtered = np.abs(np.fft.fft(self.filtered_signal)[:n//2])
        
        # Перевірка на NaN
        assert not np.isnan(self.fft_pure).any(), "NaN у fft_pure"
        assert not np.isnan(self.fft_noisy).any(), "NaN у fft_noisy"
        assert not np.isnan(self.fft_filtered).any(), "NaN у fft_filtered"
    
    def apply_custom_filter(self, signal):
        if self.filter_type == "Низькочастотний":
            return self.lowpass_filter(signal)
        elif self.filter_type == "Високочастотний":
            return self.highpass_filter(signal)
        else:
            return self.bandpass_filter(signal)
    
    def lowpass_filter(self, signal):
        filtered = np.zeros_like(signal)
        alpha = 1 / (1 + 2 * np.pi * self.cutoff_freq * (self.time[1] - self.time[0]))
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
        return filtered
    
    def highpass_filter(self, signal):
        filtered = np.zeros_like(signal)
        alpha = 1 / (1 + 2 * np.pi * self.cutoff_freq * (self.time[1] - self.time[0]))
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * filtered[i-1] + alpha * (signal[i] - signal[i-1])
        return filtered
    
    def bandpass_filter(self, signal):
        lowpass = self.lowpass_filter(signal)
        return self.highpass_filter(lowpass)
    
    def create_plots(self):
        self.time_plot = figure(title="Сигнал у часовій області", width=800, height=300, output_backend="webgl")
        self.time_plot.line('time', 'pure', source=self.source_time, color="green", legend_label="Чистий сигнал")
        self.time_plot.line('time', 'noisy', source=self.source_time, color="red", alpha=0.6, legend_label="Зашумлений")
        self.time_plot.line('time', 'filtered', source=self.source_time, color="blue", legend_label="Відфільтрований")
        self.time_plot.legend.click_policy = "hide"
        
        self.freq_plot = figure(title="Частотний спектр", width=800, height=300, output_backend="webgl")
        self.freq_plot.line('freq', 'fft_pure', source=self.source_freq, color="green", legend_label="Чистий FFT")
        self.freq_plot.line('freq', 'fft_noisy', source=self.source_freq, color="red", alpha=0.6, legend_label="Зашумлений FFT")
        self.freq_plot.line('freq', 'fft_filtered', source=self.source_freq, color="blue", legend_label="Відфільтрований FFT")
        self.freq_plot.legend.click_policy = "hide"
    
    def create_controls(self):
        self.amp_slider = Slider(title="Амплітуда", value=1.0, start=0.1, end=5.0, step=0.1)
        self.freq_slider = Slider(title="Частота (Гц)", value=1.0, start=0.1, end=10.0, step=0.1)
        self.phase_slider = Slider(title="Фаза (рад)", value=0.0, start=0, end=2*np.pi, step=0.1)
        self.noise_mean_slider = Slider(title="Середнє шуму", value=0.0, start=-1.0, end=1.0, step=0.05)
        self.noise_cov_slider = Slider(title="Дисперсія шуму", value=0.1, start=0.0, end=1.0, step=0.01)
        self.filter_select = Select(title="Тип фільтру", value="Низькочастотний", options=self.filter_types)
        self.order_slider = Slider(title="Порядок фільтру", value=4, start=1, end=10, step=1)
        self.cutoff_slider = Slider(title="Частота зрізу (Гц)", value=2.0, start=0.1, end=5.0, step=0.1)
        self.reset_btn = Button(label="Скинути", button_type="warning")
        self.update_btn = Button(label="Оновити", button_type="success")
    
    def setup_callbacks(self):
        controls = [
            self.amp_slider, self.freq_slider, self.phase_slider,
            self.noise_mean_slider, self.noise_cov_slider,
            self.filter_select, self.order_slider, self.cutoff_slider
        ]
        for control in controls:
            control.on_change('value', lambda attr, old, new: self.update_data())
        self.update_btn.on_click(self.update_data)
        self.reset_btn.on_click(self.reset)
    
    def update_data(self):
        self.amplitude = self.amp_slider.value
        self.frequency = self.freq_slider.value
        self.phase = self.phase_slider.value
        self.noise_mean = self.noise_mean_slider.value
        self.noise_cov = self.noise_cov_slider.value
        self.filter_type = self.filter_select.value
        self.filter_order = self.order_slider.value
        self.cutoff_freq = self.cutoff_slider.value
        self.generate_data()
        self.update_sources()
    
    def reset(self):
        self.amp_slider.value = 1.0
        self.freq_slider.value = 1.0
        self.phase_slider.value = 0.0
        self.noise_mean_slider.value = 0.0
        self.noise_cov_slider.value = 0.1
        self.filter_select.value = "Низькочастотний"
        self.order_slider.value = 4
        self.cutoff_slider.value = 2.0
        self.update_data()
    
    def update_sources(self):
        self.source_time.data = {
            'time': self.time,
            'pure': self.pure_signal,
            'noisy': self.noisy_signal,
            'filtered': self.filtered_signal
        }
        self.source_freq.data = {
            'freq': self.freq_axis,
            'fft_pure': self.fft_pure,
            'fft_noisy': self.fft_noisy,
            'fft_filtered': self.fft_filtered
        }

# Запуск
visualizer = HarmonicVisualizer()
layout = column(
    row(visualizer.amp_slider, visualizer.freq_slider, visualizer.phase_slider),
    row(visualizer.noise_mean_slider, visualizer.noise_cov_slider),
    row(visualizer.filter_select, visualizer.order_slider, visualizer.cutoff_slider),
    row(visualizer.reset_btn, visualizer.update_btn),
    visualizer.time_plot,
    visualizer.freq_plot
)

curdoc().add_root(layout)
curdoc().title = "Візуалізація гармоніки з фільтрацією"
