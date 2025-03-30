import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns
import re

#
def process_csv_files(data_dir):
    """Читает CSV-файлы из папки и объединяет их в один DataFrame."""
    all_data = pd.DataFrame()
    
    # 
    if not os.path.exists(data_dir):
        st.error(f"Директория {data_dir} не существует. Создаем...")
        os.makedirs(data_dir)
        return all_data
    
    #
    all_files = os.listdir(data_dir)
    st.info(f"Найдено файлов в директории: {len(all_files)}")
    st.write(f"Список файлов: {all_files}")
    
    # 
    files = [f for f in all_files if f.endswith('.csv')]
    
    if not files:
        st.warning(f"CSV-файлы не найдены в директории {data_dir}")
        return all_data
    
    st.success(f"Найдено {len(files)} CSV-файлов для обработки")
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            # 
            with open(file_path, 'r') as f:
                sample = ''.join(f.readlines(5))
                st.code(sample, language="csv")
            
            # 
            df = pd.read_csv(file_path, skiprows=2, names=["Year", "Week", "SMN", "SMT", "VCI", "TCI", "VHI", "empty"])
            
            #
            region_match = re.search(r'vhi_(\d+)', file)
            if region_match:
                df["Region_ID"] = int(region_match.group(1))
            else:
                st.warning(f"Не удалось извлечь ID региона из имени файла {file}")
                continue
                
            all_data = pd.concat([all_data, df], ignore_index=True)
            st.info(f"Файл {file} успешно обработан, добавлено {len(df)} строк")
        except Exception as e:
            st.error(f"Ошибка при чтении файла {file}: {e}")
    
    # 
    if not all_data.empty:
        all_data.drop(columns=["empty"], inplace=True, errors='ignore')
        initial_rows = len(all_data)
        all_data.dropna(subset=["VHI"], inplace=True)
        st.info(f"Удалено {initial_rows - len(all_data)} строк с отсутствующими данными VHI")
        
        # 
        all_data['Year'] = all_data['Year'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))
        all_data['Year'] = pd.to_numeric(all_data['Year'], errors='coerce')
        
        st.success(f"Итоговый DataFrame содержит {len(all_data)} строк и {len(all_data.columns)} столбцов")
    
    return all_data


# 
def main():
    st.title("Анализ вегетационного индекса здоровья (VHI)")
    
    # 
    # 
    data_dir = "D:\KPI\APD\git\Lab\Lab2\data"  # 
    
    #
    st.sidebar.header("Настройки данных")
    st.sidebar.info(f"Используется путь к данным: {os.path.abspath(data_dir)}")
    
    # 
    if os.path.exists(data_dir):
        st.sidebar.success(f"Директория {data_dir} найдена")
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if files:
            st.sidebar.success(f"Найдено {len(files)} CSV-файлов")
        else:
            st.sidebar.warning("CSV-файлы не найдены в директории")
    else:
        st.sidebar.error(f"Директория {data_dir} не найдена")
    
    # 
    st.sidebar.write(f"Текущая директория: {os.getcwd()}")
    
    st.sidebar.header("Опции фильтрации")
    
    # 
    # 
    if os.path.exists(data_dir):
        with st.spinner("Загружаем данные..."):
            df = process_csv_files(data_dir)
        
        if not df.empty:
            st.session_state['data'] = df
            st.success("Данные успешно загружены!")
            # 
            st.write("Первые 5 строк данных:")
            st.write(df.head())
        else:
            st.error("Не удалось загрузить данные. Пустой DataFrame.")
    else:
        st.error(f"Директория {data_dir} не найдена")
        st.info("Создайте директорию 'data' и поместите в неё CSV-файлы с данными")
    
    # 
    if st.sidebar.button("Перезагрузить данные"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    #
    if 'data' in st.session_state and not st.session_state['data'].empty:
        df = st.session_state['data']
        
        #
        st.header("Анализ данных")
        
        # 
        regions = sorted(df['Region_ID'].unique())
        years = sorted(df['Year'].unique())
        
        selected_region = st.selectbox("Выберите регион", regions)
        selected_year = st.selectbox("Выберите год", years)
        
        # 
        filtered_data = df[(df['Region_ID'] == selected_region) & (df['Year'] == selected_year)]
        
        if not filtered_data.empty:
            st.subheader(f"Данные для региона {selected_region}, год {selected_year}")
            st.write(filtered_data)
            
            # 
            st.subheader("Визуализация VHI по неделям")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_data['Week'], filtered_data['VHI'], marker='o')
            ax.set_xlabel('Неделя')
            ax.set_ylabel('VHI')
            ax.set_title(f'VHI для региона {selected_region}, год {selected_year}')
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning(f"Нет данных для региона {selected_region} за год {selected_year}")
    else:
        st.warning("Отсутствуют CSV-файлы с данными. Пожалуйста, добавьте файлы в папку 'data'.")

if __name__ == "__main__":
    main()