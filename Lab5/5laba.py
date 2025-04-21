import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy import signal
import time

class HarmonicVisualizer:
    def __init__(self):
        # Початкові параметри
        self.time_points = np.linspace(0, 10, 1000)  # Часові точки від 0 до 10 секунд
        
        # Початкові параметри гармоніки
        self.init_amplitude = 1.0  # Початкова амплітуда
        self.init_frequency = 1.0  # Початкова частота (Гц)
        self.init_phase = 0.0      # Початковий фазовий зсув (радіани)
        
        # Початкові параметри шуму
        self.init_noise_mean = 0.0      # Середнє значення шуму
        self.init_noise_covariance = 0.1  # Дисперсія шуму
        
        # Поточні параметри
        self.amplitude = self.init_amplitude
        self.frequency = self.init_frequency
        self.phase = self.init_phase
        self.noise_mean = self.init_noise_mean
        self.noise_covariance = self.init_noise_covariance
        
        # Флаги відображення
        self.show_noise = True
        self.show_filtered = True
        
        # Параметри фільтру
        self.filter_order = 4
        self.filter_cutoff = 2.0  # Частота зрізу (Гц)
        
        # Генерація початкових даних
        self.pure_harmonic = self.generate_pure_harmonic()
        self.noise = self.generate_noise()
        self.noisy_harmonic = self.pure_harmonic + self.noise
        self.filtered_harmonic = self.apply_filter(self.noisy_harmonic)
        
        # Створення вікна та осей
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.subplots_adjust(left=0.25, bottom=0.35)
        
        # Ініціалізація графіків
        self.pure_line, = self.ax1.plot(self.time_points, self.pure_harmonic, 'g-', lw=2, label='Чиста гармоніка')
        self.noisy_line, = self.ax1.plot(self.time_points, self.noisy_harmonic, 'r-', lw=1, alpha=0.7, label='Гармоніка з шумом')
        
        # Графік для порівняння фільтрованого сигналу
        self.pure_line2, = self.ax2.plot(self.time_points, self.pure_harmonic, 'g-', lw=2, label='Чиста гармоніка')
        self.filtered_line, = self.ax2.plot(self.time_points, self.filtered_harmonic, 'b-', lw=1.5, label='Відфільтрована гармоніка')
        
        # Налаштування осей
        self.ax1.set_title('Гармоніка та гармоніка з шумом')
        self.ax1.set_xlabel('Час (с)')
        self.ax1.set_ylabel('Амплітуда')
        self.ax1.grid(True)
        self.ax1.legend(loc='upper right')
        
        self.ax2.set_title('Порівняння чистої та відфільтрованої гармоніки')
        self.ax2.set_xlabel('Час (с)')
        self.ax2.set_ylabel('Амплітуда')
        self.ax2.grid(True)
        self.ax2.legend(loc='upper right')
        
        # Створення слайдерів для параметрів гармоніки
        self.axamp = plt.axes([0.25, 0.20, 0.65, 0.03])
        self.axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.axphase = plt.axes([0.25, 0.10, 0.65, 0.03])
        
        self.amp_slider = Slider(self.axamp, 'Амплітуда', 0.1, 2.0, valinit=self.init_amplitude)
        self.freq_slider = Slider(self.axfreq, 'Частота', 0.1, 5.0, valinit=self.init_frequency)
        self.phase_slider = Slider(self.axphase, 'Фаза', 0.0, 2*np.pi, valinit=self.init_phase)
        
        # Створення слайдерів для параметрів шуму
        self.axnoise_mean = plt.axes([0.25, 0.05, 0.30, 0.03])
        self.axnoise_cov = plt.axes([0.25, 0.01, 0.30, 0.03])
        
        self.noise_mean_slider = Slider(self.axnoise_mean, 'Середнє шуму', -0.5, 0.5, valinit=self.init_noise_mean)
        self.noise_cov_slider = Slider(self.axnoise_cov, 'Дисперсія шуму', 0.0, 0.5, valinit=self.init_noise_covariance)
        
        # Створення слайдерів для параметрів фільтру
        self.axfilter_order = plt.axes([0.6, 0.05, 0.30, 0.03])
        self.axfilter_cutoff = plt.axes([0.6, 0.01, 0.30, 0.03])
        
        self.filter_order_slider = Slider(self.axfilter_order, 'Порядок фільтру', 1, 10, valinit=self.filter_order, valstep=1)
        self.filter_cutoff_slider = Slider(self.axfilter_cutoff, 'Частота зрізу', 0.1, 10.0, valinit=self.filter_cutoff)
        
        # Створення кнопки Reset
        self.resetax = plt.axes([0.8, 0.25, 0.1, 0.04])
        self.button = Button(self.resetax, 'Reset')
        
        # Створення чекбоксів
        self.checkax = plt.axes([0.05, 0.4, 0.15, 0.15])
        self.check = CheckButtons(
            self.checkax, ['Показати шум', 'Показати фільтр'], [True, True]
        )
        
        # Підключення колбеків
        self.amp_slider.on_changed(self.update_harmonic)
        self.freq_slider.on_changed(self.update_harmonic)
        self.phase_slider.on_changed(self.update_harmonic)
        self.noise_mean_slider.on_changed(self.update_noise)
        self.noise_cov_slider.on_changed(self.update_noise)
        self.filter_order_slider.on_changed(self.update_filter)
        self.filter_cutoff_slider.on_changed(self.update_filter)
        self.button.on_clicked(self.reset)
        self.check.on_clicked(self.toggle_display)
        
    def generate_pure_harmonic(self):
        """Генерація чистої гармоніки з поточними параметрами"""
        return self.amplitude * np.sin(2 * np.pi * self.frequency * self.time_points + self.phase)
    
    def generate_noise(self):
        """Генерація шуму з поточними параметрами"""
        return np.random.normal(self.noise_mean, np.sqrt(self.noise_covariance), len(self.time_points))
    
    def apply_filter(self, signal_data):
        """Застосування фільтру до сигналу"""
        # Фільтр нижніх частот Butterworth
        b, a = signal.butter(self.filter_order, self.filter_cutoff, 'low', fs=len(self.time_points)/self.time_points[-1])
        return signal.filtfilt(b, a, signal_data)
    
    def update_harmonic(self, val):
        """Оновлення параметрів гармоніки"""
        self.amplitude = self.amp_slider.val
        self.frequency = self.freq_slider.val
        self.phase = self.phase_slider.val
        
        # Оновлення чистої гармоніки
        self.pure_harmonic = self.generate_pure_harmonic()
        
        # Оновлення зашумленої гармоніки (шум залишається тим самим)
        self.noisy_harmonic = self.pure_harmonic + self.noise
        
        # Оновлення відфільтрованої гармоніки
        self.filtered_harmonic = self.apply_filter(self.noisy_harmonic)
        
        # Оновлення графіків
        self.pure_line.set_ydata(self.pure_harmonic)
        self.noisy_line.set_ydata(self.noisy_harmonic)
        self.pure_line2.set_ydata(self.pure_harmonic)
        self.filtered_line.set_ydata(self.filtered_harmonic)
        
        # Оновлення меж осей
        self._update_axes_limits()
        
        self.fig.canvas.draw_idle()
    
    def update_noise(self, val):
        """Оновлення параметрів шуму"""
        self.noise_mean = self.noise_mean_slider.val
        self.noise_covariance = self.noise_cov_slider.val
        
        # Оновлення лише шуму, гармоніка залишається тією самою
        self.noise = self.generate_noise()
        self.noisy_harmonic = self.pure_harmonic + self.noise
        
        # Оновлення відфільтрованої гармоніки
        self.filtered_harmonic = self.apply_filter(self.noisy_harmonic)
        
        # Оновлення графіків
        self.noisy_line.set_ydata(self.noisy_harmonic)
        self.filtered_line.set_ydata(self.filtered_harmonic)
        
        # Оновлення меж осей
        self._update_axes_limits()
        
        self.fig.canvas.draw_idle()
    
    def update_filter(self, val):
        """Оновлення параметрів фільтру"""
        self.filter_order = int(self.filter_order_slider.val)
        self.filter_cutoff = self.filter_cutoff_slider.val
        
        # Оновлення лише відфільтрованої гармоніки
        self.filtered_harmonic = self.apply_filter(self.noisy_harmonic)
        
        # Оновлення графіка
        self.filtered_line.set_ydata(self.filtered_harmonic)
        
        self.fig.canvas.draw_idle()
    
    def reset(self, event):
        """Відновлення початкових параметрів"""
        # Відновлення слайдерів
        self.amp_slider.reset()
        self.freq_slider.reset()
        self.phase_slider.reset()
        self.noise_mean_slider.reset()
        self.noise_cov_slider.reset()
        self.filter_order_slider.reset()
        self.filter_cutoff_slider.reset()
        
        # Відновлення параметрів
        self.amplitude = self.init_amplitude
        self.frequency = self.init_frequency
        self.phase = self.init_phase
        self.noise_mean = self.init_noise_mean
        self.noise_covariance = self.init_noise_covariance
        self.filter_order = 4
        self.filter_cutoff = 2.0
        
        # Відновлення чекбоксів
        self.check.set_active(0)
        self.check.set_active(1)
        self.show_noise = True
        self.show_filtered = True
        
        # Повторне генерування даних
        self.pure_harmonic = self.generate_pure_harmonic()
        self.noise = self.generate_noise()
        self.noisy_harmonic = self.pure_harmonic + self.noise
        self.filtered_harmonic = self.apply_filter(self.noisy_harmonic)
        
        # Оновлення графіків
        self.pure_line.set_ydata(self.pure_harmonic)
        self.noisy_line.set_ydata(self.noisy_harmonic)
        self.noisy_line.set_visible(True)
        self.pure_line2.set_ydata(self.pure_harmonic)
        self.filtered_line.set_ydata(self.filtered_harmonic)
        self.filtered_line.set_visible(True)
        
        # Оновлення меж осей
        self._update_axes_limits()
        
        self.fig.canvas.draw_idle()
    
    def toggle_display(self, label):
        """Перемикання відображення шуму та фільтрованого сигналу"""
        if label == 'Показати шум':
            self.show_noise = not self.show_noise
            self.noisy_line.set_visible(self.show_noise)
        elif label == 'Показати фільтр':
            self.show_filtered = not self.show_filtered
            self.filtered_line.set_visible(self.show_filtered)
        
        self.fig.canvas.draw_idle()
    
    def _update_axes_limits(self):
        """Оновлення меж осей графіків"""
        # Обчислення меж для першого графіка
        y_max1 = max(np.max(self.pure_harmonic), np.max(self.noisy_harmonic if self.show_noise else [0]))
        y_min1 = min(np.min(self.pure_harmonic), np.min(self.noisy_harmonic if self.show_noise else [0]))
        margin1 = 0.1 * (y_max1 - y_min1)
        self.ax1.set_ylim(y_min1 - margin1, y_max1 + margin1)
        
        # Обчислення меж для другого графіка
        y_max2 = max(np.max(self.pure_harmonic), np.max(self.filtered_harmonic if self.show_filtered else [0]))
        y_min2 = min(np.min(self.pure_harmonic), np.min(self.filtered_harmonic if self.show_filtered else [0]))
        margin2 = 0.1 * (y_max2 - y_min2)
        self.ax2.set_ylim(y_min2 - margin2, y_max2 + margin2)
    
    def show(self):
        """Показ графічного інтерфейсу"""
        plt.show()


# Створення та запуск візуалізатора
def main():
    print("Запуск візуалізатора гармонік...")
    print("Інтерактивні елементи управління дозволяють змінювати параметри гармоніки, шуму та фільтрації.")
    print("Використовуйте слайдери для налаштування параметрів та спостерігайте за змінами на графіках.")
    print("Чисту гармоніку зображено зеленим кольором, гармоніку з шумом - червоним, відфільтровану - синім.")
    
    visualizer = HarmonicVisualizer()
    visualizer.show()

if __name__ == "__main__":
    main()