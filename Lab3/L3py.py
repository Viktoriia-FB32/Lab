import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import re

def process_csv_files(data_dir):
    all_data = pd.DataFrame()
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return all_data
    
    all_files = os.listdir(data_dir)
    files = [f for f in all_files if f.endswith('.csv')]
    
    if not files:
        return all_data
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path, skiprows=2, names=["Year", "Week", "SMN", "SMT", "VCI", "TCI", "VHI", "empty"])
            
            region_match = re.search(r'vhi_(\d+)', file)
            if region_match:
                df["Region_ID"] = int(region_match.group(1))
            else:
                continue
                
            all_data = pd.concat([all_data, df], ignore_index=True)
        except Exception as e:
            pass
    
    if not all_data.empty:
        all_data.drop(columns=["empty"], inplace=True, errors='ignore')
        all_data.dropna(subset=["VHI"], inplace=True)
        
        all_data['Year'] = all_data['Year'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))
        all_data['Year'] = pd.to_numeric(all_data['Year'], errors='coerce')
    
    return all_data

def main():
    st.set_page_config(layout="wide")
    st.title("Аналіз вегетаційного індексу здоров’я (VHI)")
    
    data_dir = "D:\KPI\APD\git\Lab\Lab2\data"
    
    # Автоматичне завантаження даних при запуску
    if os.path.exists(data_dir):
        with st.spinner("Завантажуємо дані..."):
            df = process_csv_files(data_dir)
        
        if not df.empty:
            st.session_state['data'] = df
        else:
            st.error("Не вдалося завантажити дані. DataFrame порожній.")
    
    # Перевіряємо, чи завантажені дані
    if 'data' in st.session_state and not st.session_state['data'].empty:
        df = st.session_state['data']
        
        # Створюємо двоколонковий макет
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.header("Параметри фільтрації")
            
            # 1. Dropdown для вибору індексу
            index_option = st.selectbox(
                "Оберіть індекс для аналізу",
                ["VCI", "TCI", "VHI"],
                key="index_option"
            )
            
            # 2. Dropdown для вибору області
            regions = sorted(df['Region_ID'].unique())
            selected_region = st.selectbox(
                "Оберіть область",
                regions,
                key="selected_region"
            )
            
            # 3. Slider для інтервалу тижнів
            min_week, max_week = int(df['Week'].min()), int(df['Week'].max())
            weeks_range = st.slider(
                "Оберіть інтервал тижнів",
                min_week, max_week, (min_week, max_week),
                key="weeks_range"
            )
            
            # 4. Slider для інтервалу років
            years = sorted(df['Year'].unique())
            years = [y for y in years if not pd.isna(y)]
            min_year, max_year = int(min(years)), int(max(years))
            years_range = st.slider(
                "Оберіть інтервал років",
                min_year, max_year, (min_year, max_year),
                key="years_range"
            )
            
            # 8. Checkboxes для сортування
            st.subheader("Сортування даних")
            sort_asc = st.checkbox("Сортувати за зростанням", key="sort_asc")
            sort_desc = st.checkbox("Сортувати за спаданням", key="sort_desc")
            
            # 5. Кнопка для скидання фільтрів
            if st.button("Скинути всі фільтри"):
                st.session_state["index_option"] = "VHI"
                st.session_state["selected_region"] = regions[0]
                st.session_state["weeks_range"] = (min_week, max_week)
                st.session_state["years_range"] = (min_year, max_year)
                st.session_state["sort_asc"] = False
                st.session_state["sort_desc"] = False
                st.rerun()
        
        with col2:
            # Фільтрація даних на основі вибраних параметрів
            filtered_data = df[
                (df['Region_ID'] == selected_region) & 
                (df['Year'] >= years_range[0]) & 
                (df['Year'] <= years_range[1]) &
                (df['Week'] >= weeks_range[0]) & 
                (df['Week'] <= weeks_range[1])
            ]
            
            # Сортування даних (якщо обрано обидва чекбокси — пріоритет за зростанням)
            if sort_asc and not sort_desc:
                filtered_data = filtered_data.sort_values(by=index_option, ascending=True)
            elif sort_desc and not sort_asc:
                filtered_data = filtered_data.sort_values(by=index_option, ascending=False)
            elif sort_asc and sort_desc:
                st.warning("Обрано обидві опції сортування. Застосовується сортування за зростанням.")
                filtered_data = filtered_data.sort_values(by=index_option, ascending=True)
            
            # 6. Створення вкладок для таблиці та графіків
            tab1, tab2, tab3 = st.tabs(["Таблиця даних", "Графік динаміки", "Порівняння областей"])
            
            with tab1:
                st.subheader(f"Таблиця даних для області {selected_region}")
                st.dataframe(filtered_data)
            
            with tab2:
                st.subheader(f"Динаміка {index_option} для області {selected_region}")
                
                if not filtered_data.empty:
                    # Групуємо дані за роком і тижнем для наочності
                    pivoted_data = filtered_data.pivot_table(
                        index='Week', 
                        columns='Year', 
                        values=index_option,
                        aggfunc='mean'
                    )
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    pivoted_data.plot(ax=ax, marker='o', linestyle='-')
                    ax.set_xlabel('Тиждень')
                    ax.set_ylabel(index_option)
                    ax.set_title(f'{index_option} для області {selected_region} ({years_range[0]}-{years_range[1]})')
                    ax.grid(True)
                    ax.legend(title='Рік')
                    st.pyplot(fig)
                else:
                    st.warning(f"Немає даних для області {selected_region} за вибраний період")
            
            with tab3:
                st.subheader(f"Порівняння  {index_option} по областях")
                
                # Отримуємо дані по всіх областях за вибраний часовий інтервал
                all_regions_data = df[
                    (df['Year'] >= years_range[0]) & 
                    (df['Year'] <= years_range[1]) &
                    (df['Week'] >= weeks_range[0]) & 
                    (df['Week'] <= weeks_range[1])
                ]
                
                if not all_regions_data.empty:
                    # Розраховуємо середні значення індексу для кожної області
                    region_averages = all_regions_data.groupby('Region_ID')[index_option].mean().reset_index()
                    
                    # Виділяємо вибрану область
                    region_averages['Highlighted'] = region_averages['Region_ID'] == selected_region
                    
                    # Створюємо графік
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Сортуємо дані за значенням індексу для кращої візуалізації
                    region_averages = region_averages.sort_values(by=index_option)
                    
                    # Створюємо кольорову мапу: вибрана область — червона, інші — сірі
                    colors = ['red' if highlighted else 'gray' for highlighted in region_averages['Highlighted']]
                    
                    # Додаємо підписи та сітку
                    bars = ax.bar(region_averages['Region_ID'].astype(str), region_averages[index_option], color=colors)
                    
                    # Добавляем подписи и сетку
                    ax.set_xlabel('Область  (ID)')
                    ax.set_ylabel(f'Середнє значення {index_option}')
                    ax.set_title(f'Порівняння середнього {index_option} по областям ({years_range[0]}-{years_range[1]})')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Додаємо анотацію для вибраної області
                    for i, bar in enumerate(bars):
                        if region_averages.iloc[i]['Highlighted']:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f"Область {selected_region}",
                                ha='center', va='bottom', rotation=0,
                                color='red', fontweight='bold')

                    
                    st.pyplot(fig)
                    
                    # Лінійний графік для порівняння динаміки вибраної області зі середнім по всіх
                    st.subheader(f"Динаміка {index_option} по роках: область {selected_region} vs середнє")
                    
                    # Обчислюємо середні значення по роках 
                    selected_region_yearly = all_regions_data[all_regions_data['Region_ID'] == selected_region].groupby('Year')[index_option].mean()
                    all_regions_yearly = all_regions_data.groupby('Year')[index_option].mean()
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    selected_region_yearly.plot(ax=ax2, marker='o', linestyle='-', color='red', label=f'Область {selected_region}')
                    all_regions_yearly.plot(ax=ax2, marker='s', linestyle='--', color='blue', label='Середнє по всім областям')
                    
                    ax2.set_xlabel('Рік')
                    ax2.set_ylabel(index_option)
                    ax2.set_title(f'Порівняння динаміки {index_option} по рокам')
                    ax2.grid(True)
                    ax2.legend()
                    
                    st.pyplot(fig2)
                else:
                    st.warning("Немає даних за вибраний період")
    else:
        st.warning("Дані не завантажено. Перевірте наявність файлів у директорії 'data'.")

if __name__ == "__main__":
    main()