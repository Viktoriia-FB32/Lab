import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import timeit
import time
from scipy import stats

# Шлях до файлу з даними - змініть на свій локальний шлях
file_path = "household_power_consumption.txt"

# Функція для виведення результатів профілювання
def print_timing_results(pandas_time, numpy_time, operation_name):
    print(f"\n--- Профілювання часу: {operation_name} ---")
    print(f"Pandas: {pandas_time:.6f} секунд")
    print(f"NumPy: {numpy_time:.6f} секунд")
    if pandas_time < numpy_time:
        print(f"Pandas швидший на {(numpy_time/pandas_time - 1)*100:.2f}%")
    else:
        print(f"NumPy швидший на {(pandas_time/numpy_time - 1)*100:.2f}%")

# Завантаження даних з використанням Pandas
def load_data_pandas():
    # Визначення типів даних для оптимізації пам'яті
    dtypes = {
        'Global_active_power': 'float64',
        'Global_reactive_power': 'float64',
        'Voltage': 'float64',
        'Global_intensity': 'float64',
        'Sub_metering_1': 'float64',
        'Sub_metering_2': 'float64',
        'Sub_metering_3': 'float64'
    }
    
    # Завантаження даних з визначенням роздільника та пропущених значень
    df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']},
                    dayfirst=True, na_values=['?'], dtype=dtypes)
    
    # Видалення рядків з пропущеними значеннями
    df = df.dropna()
    
    # Додавання окремих стовпців для дати і часу для зручності обробки
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['hour'] = df['datetime'].dt.hour
    
    return df

# Завантаження даних з використанням NumPy
def load_data_numpy():
    # Визначення типів даних
    types = [("Date", "U10"), ("Time", "U8"), ("Global_active_power", "float64"),
             ("Global_reactive_power", "float64"), ("Voltage", "float64"),
             ("Global_intensity", "float64"), ("Sub_metering_1", "float64"),
             ("Sub_metering_2", "float64"), ("Sub_metering_3", "float64")]
    
    # Завантаження даних
    data = np.genfromtxt(file_path, missing_values=["?", np.nan],
                         delimiter=';', dtype=types, encoding="UTF-8", names=True)
    
    # Видалення рядків з пропущеними значеннями
    mask = ~np.isnan(data["Global_active_power"])
    data = data[mask]
    
    return data

# Завдання 1: Обрати всі записи, у яких загальна активна споживана потужність перевищує 5 кВт
def task1_pandas(df):
    result = df[df['Global_active_power'] > 5]
    return result

def task1_numpy(data):
    mask = data["Global_active_power"] > 5
    result = data[mask]
    return result

# Завдання 2: Обрати всі записи, у яких вольтаж перевищую 235 В
def task2_pandas(df):
    result = df[df['Voltage'] > 235]
    return result

def task2_numpy(data):
    mask = data["Voltage"] > 235
    result = data[mask]
    return result

# Завдання 3: Обрати всі записи, у яких сила струму лежить в межах 19-20 А,
# для них виявити ті, у яких пральна машина та холодильних споживають більше, ніж бойлер та кондиціонер
def task3_pandas(df):
    # Спочатку фільтруємо за силою струму
    current_filter = (df['Global_intensity'] >= 19) & (df['Global_intensity'] <= 20)
    current_filtered = df[current_filter]
    
    # Потім фільтруємо за умовою споживання
    result = current_filtered[current_filtered['Sub_metering_2'] > current_filtered['Sub_metering_3']]
    return result

def task3_numpy(data):
    # Спочатку фільтруємо за силою струму
    current_mask = (data["Global_intensity"] >= 19) & (data["Global_intensity"] <= 20)
    current_filtered = data[current_mask]
    
    # Потім фільтруємо за умовою споживання
    consumption_mask = current_filtered["Sub_metering_2"] > current_filtered["Sub_metering_3"]
    result = current_filtered[consumption_mask]
    return result

# Завдання 4: Обрати випадковим чином 500000 записів (без повторів елементів вибірки),
# для них обчислити середні величини усіх 3-х груп споживання електричної енергії
def task4_pandas(df):
    if len(df) >= 500000:
        # Обираємо випадковим чином 500000 записів без повторень
        sample = df.sample(n=500000, replace=False)
        
        # Обчислюємо середні величини для 3-х груп споживання
        means = {
            'Sub_metering_1_mean': sample['Sub_metering_1'].mean(),
            'Sub_metering_2_mean': sample['Sub_metering_2'].mean(),
            'Sub_metering_3_mean': sample['Sub_metering_3'].mean()
        }
    else:
        # Якщо даних менше ніж 500000, використовуємо всі дані
        sample = df
        means = {
            'Sub_metering_1_mean': sample['Sub_metering_1'].mean(),
            'Sub_metering_2_mean': sample['Sub_metering_2'].mean(),
            'Sub_metering_3_mean': sample['Sub_metering_3'].mean()
        }
    
    return means

def task4_numpy(data):
    if len(data) >= 500000:
        # Генеруємо випадкові індекси без повторень
        indices = np.random.choice(len(data), size=500000, replace=False)
        sample = data[indices]
    else:
        # Якщо даних менше ніж 500000, використовуємо всі дані
        sample = data
    
    # Обчислюємо середні величини для 3-х груп споживання
    means = {
        'Sub_metering_1_mean': np.mean(sample['Sub_metering_1']),
        'Sub_metering_2_mean': np.mean(sample['Sub_metering_2']),
        'Sub_metering_3_mean': np.mean(sample['Sub_metering_3'])
    }
    
    return means

# Завдання 5: Обрати ті записи, які після 18-00 споживають понад 6 кВт за хвилину в середньому, 
# серед відібраних визначити ті, у яких основне споживання електроенергії у вказаний проміжок часу
# припадає на пральну машину, сушарку, холодильник та освітлення (група 2 є найбільшою), 
# а потім обрати кожен третій результат із першої половини та кожен четвертий результат із другої половини.
def task5_pandas(df):
    # Фільтруємо записи після 18:00 з споживанням більше 6 кВт
    evening_high_consumption = df[(df['hour'] >= 18) & (df['Global_active_power'] > 6)]
    
    # Визначаємо записи, де група 2 споживає найбільше
    group2_highest = evening_high_consumption[
        (evening_high_consumption['Sub_metering_2'] > evening_high_consumption['Sub_metering_1']) & 
        (evening_high_consumption['Sub_metering_2'] > evening_high_consumption['Sub_metering_3'])
    ]
    
    # Розділяємо на дві половини
    half_point = len(group2_highest) // 2
    first_half = group2_highest.iloc[:half_point]
    second_half = group2_highest.iloc[half_point:]
    
    # Обираємо кожен третій результат з першої половини
    first_half_selection = first_half.iloc[::3]
    
    # Обираємо кожен четвертий результат з другої половини
    second_half_selection = second_half.iloc[::4]
    
    # Об'єднуємо результати
    result = pd.concat([first_half_selection, second_half_selection])
    
    return result

def task5_numpy(data):
    # Витягаємо години з часу (припустимо, що час у форматі "HH:MM:SS")
    hours = np.array([int(time.split(':')[0]) for time in data['Time']])
    
    # Фільтруємо записи після 18:00 з споживанням більше 6 кВт
    evening_mask = (hours >= 18) & (data['Global_active_power'] > 6)
    evening_high_consumption = data[evening_mask]
    
    # Визначаємо записи, де група 2 споживає найбільше
    group2_mask = (evening_high_consumption['Sub_metering_2'] > evening_high_consumption['Sub_metering_1']) & \
                  (evening_high_consumption['Sub_metering_2'] > evening_high_consumption['Sub_metering_3'])
    group2_highest = evening_high_consumption[group2_mask]
    
    # Розділяємо на дві половини
    half_point = len(group2_highest) // 2
    first_half = group2_highest[:half_point]
    second_half = group2_highest[half_point:]
    
    # Обираємо кожен третій результат з першої половини
    first_half_selection = first_half[::3]
    
    # Обираємо кожен четвертий результат з другої половини
    second_half_selection = second_half[::4]
    
    # Об'єднуємо результати (для NumPy потрібно використовувати np.concatenate)
    result = np.concatenate([first_half_selection, second_half_selection])
    
    return result

# Функції для виконання і профілювання всіх завдань
def run_all_tasks_pandas(df):
    results = {}
    
    task1_time = timeit(lambda: task1_pandas(df), number=5) / 5
    results['task1'] = {'time': task1_time, 'result_len': len(task1_pandas(df))}
    
    task2_time = timeit(lambda: task2_pandas(df), number=5) / 5
    results['task2'] = {'time': task2_time, 'result_len': len(task2_pandas(df))}
    
    task3_time = timeit(lambda: task3_pandas(df), number=5) / 5
    results['task3'] = {'time': task3_time, 'result_len': len(task3_pandas(df))}
    
    task4_time = timeit(lambda: task4_pandas(df), number=5) / 5
    results['task4'] = {'time': task4_time, 'result': task4_pandas(df)}
    
    task5_time = timeit(lambda: task5_pandas(df), number=5) / 5
    try:
        results['task5'] = {'time': task5_time, 'result_len': len(task5_pandas(df))}
    except Exception as e:
        results['task5'] = {'time': task5_time, 'error': str(e)}
    
    return results

def run_all_tasks_numpy(data):
    results = {}
    
    task1_time = timeit(lambda: task1_numpy(data), number=5) / 5
    results['task1'] = {'time': task1_time, 'result_len': len(task1_numpy(data))}
    
    task2_time = timeit(lambda: task2_numpy(data), number=5) / 5
    results['task2'] = {'time': task2_time, 'result_len': len(task2_numpy(data))}
    
    task3_time = timeit(lambda: task3_numpy(data), number=5) / 5
    results['task3'] = {'time': task3_time, 'result_len': len(task3_numpy(data))}
    
    task4_time = timeit(lambda: task4_numpy(data), number=5) / 5
    results['task4'] = {'time': task4_time, 'result': task4_numpy(data)}
    
    task5_time = timeit(lambda: task5_numpy(data), number=5) / 5
    try:
        results['task5'] = {'time': task5_time, 'result_len': len(task5_numpy(data))}
    except Exception as e:
        results['task5'] = {'time': task5_time, 'error': str(e)}
    
    return results

# Візуалізація результатів порівняння часу виконання
def plot_timing_comparison(pandas_results, numpy_results):
    tasks = list(pandas_results.keys())
    pandas_times = [pandas_results[task]['time'] for task in tasks]
    numpy_times = [numpy_results[task]['time'] for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, pandas_times, width, label='Pandas')
    rects2 = ax.bar(x + width/2, numpy_times, width, label='NumPy')
    
    ax.set_ylabel('Час виконання (секунди)')
    ax.set_title('Порівняння часу виконання завдань')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Завдання {i+1}' for i in range(len(tasks))])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('timing_comparison.png')
    plt.show()

# Основна функція для виконання всіх завдань і профілювання
def main():
    print("Завантаження даних за допомогою Pandas...")
    pandas_load_time = timeit(load_data_pandas, number=1)
    df = load_data_pandas()
    
    print("Завантаження даних за допомогою NumPy...")
    numpy_load_time = timeit(load_data_numpy, number=1)
    data = load_data_numpy()
    
    print_timing_results(pandas_load_time, numpy_load_time, "Завантаження даних")
    
    print("\nВиконання завдань з використанням Pandas...")
    pandas_results = run_all_tasks_pandas(df)
    
    print("\nВиконання завдань з використанням NumPy...")
    numpy_results = run_all_tasks_numpy(data)
    
    # Виведення результатів профілювання для кожного завдання
    for i, task in enumerate(pandas_results.keys(), 1):
        print_timing_results(pandas_results[task]['time'], numpy_results[task]['time'], f"Завдання {i}")
        print(f"Розмір результату для Pandas: {pandas_results[task].get('result_len', '-')}")
        print(f"Розмір результату для NumPy: {numpy_results[task].get('result_len', '-')}")
        if 'error' in pandas_results[task]:
            print(f"Помилка при виконанні з Pandas: {pandas_results[task]['error']}")
        if 'error' in numpy_results[task]:
            print(f"Помилка при виконанні з NumPy: {numpy_results[task]['error']}")
    
    # Візуалізація порівняння часу виконання
    plot_timing_comparison(pandas_results, numpy_results)
    
    # Оцінка зручності виконання операцій
    print("\n--- Оцінка зручності виконання операцій (за 5-бальною шкалою) ---")
    print("Pandas:")
    print("Завантаження даних: 5 - легко завантажувати дані, вбудована підтримка CSV і інших форматів")
    print("Фільтрація даних: 5 - інтуїтивний синтаксис, легка робота з умовами")
    print("Агрегація даних: 5 - зручні функції для обчислення статистик")
    print("Вибірка за індексами: 5 - зручні функції loc і iloc")
    print("Робота з часовими даними: 5 - вбудована підтримка datetime")
    
    print("\nNumPy:")
    print("Завантаження даних: 3 - потребує більше коду для конфігурації")
    print("Фільтрація даних: 4 - працює швидко, але синтаксис менш інтуїтивний")
    print("Агрегація даних: 4 - хороші функції, але не так зручно як у Pandas")
    print("Вибірка за індексами: 4 - прямий доступ до елементів, але складніше для складних вибірок")
    print("Робота з часовими даними: 2 - обмежена підтримка, потребує додаткового коду")
    
    print("\n--- ВИСНОВКИ ---")
    print("1. Pandas є більш зручним для роботи з табличними даними, особливо якщо потрібна робота з датами/часом.")
    print("2. NumPy часто швидше працює з числовими операціями, але вимагає більше коду для налаштування.")
    print("3. Для обробки великих наборів даних NumPy може бути ефективнішим з точки зору використання пам'яті.")
    print("4. Pandas краще підходить для аналізу даних, візуалізації та коли потрібні складні індексування і агрегації.")
    print("5. Вибір структури даних залежить від конкретної задачі та розміру набору даних.")

if __name__ == "__main__":
    main()