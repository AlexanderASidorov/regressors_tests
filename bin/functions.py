# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error # методы оценки ошибки модели


# функция для генерации массива данных для x, y, z
def generate_random_array(n, m, low, high, seed=42):
    """
    Генерирует массив размером (n, m) со случайными вещественными числами
    в диапазоне [low, high).
    
    Параметры:
        n (int): количество строк
        m (int): количество столбцов
        low (float): нижняя граница диапазона (включительно)
        high (float): верхняя граница диапазона (исключительно)
        seed (int): для воспроизводимости
    
    Возвращает:
        np.ndarray: массив размером (n, m) типа float
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, m))


# Функция для сравнения тестовых и предсказанных значений
def plot_true_vs_predicted(y_true, y_pred, title="True vs Predicted", figsize=(8, 6)):
    """
    Строит график истинных значений vs предсказанных.
    
    Параметры:
        y_true (array-like): Истинные значения
        y_pred (array-like): Предсказанные значения
        title (str): Заголовок графика
        figsize (tuple): Размер графика
    """
    # Вычисляем метрики
    r2 = r2_score(y_true, y_pred) # коэффициент детерминации
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) #  среднеквадратическая ошибка
    mape = mean_absolute_percentage_error(y_true, y_pred) # cредняя абсолютная процентная ошибка

    # Строим график
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Ideal line')

    # Добавляем метрики на график
    metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAPE = {mape*100:.3f}%'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))

    # Оформление графика
    plt.title(title, fontsize=14)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()