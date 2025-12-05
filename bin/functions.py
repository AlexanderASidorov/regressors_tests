# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error # методы оценки ошибки модели
from xgboost import XGBRegressor
import itertools


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





class Generate_data():
    '''
    Функции для создания искуственных массивов признаков
    '''
    
    def __init__(self, n, m, low, high, seed=42):
        '''
        Параметры:
        n (int): количество точек (экспериментов)
        m (int): количество факторов (признаков)
        low (float или array-like): нижняя граница для каждого признака.
                            Если скаляр — применяется ко всем признакам.
        high (float или array-like): верхняя граница для каждого признака.
                            Если скаляр — применяется ко всем признакам.
        seed (int): для воспроизводимости
        '''
        self.n = n
        self.m = m
        self.low = low
        self.high = high
        self.seed = seed
        self.features = None
        
    
    # функция для генерации массива данных для x, y, z
    def generate_random_array(self):
        """
        Генерирует массив размером (n, m) со случайными вещественными числами
        в диапазоне [low, high).
        
               
        Возвращает:
            np.ndarray: массив размером (n, m) типа float
        """
        rng = np.random.default_rng(self.seed)
        
        self.features = rng.uniform(self.low, self.high, size=(self.n, self.m))
        
        return self.features
    
    # функция для генерации массива в виде латинсского гипер куба
    def generate_latin_hypercube(self):
        
        rng = np.random.default_rng(self.seed)
        
        low = np.full(self.m, self.low) if np.isscalar(self.low) else np.asarray(self.low)
        high = np.full(self.m, self.high) if np.isscalar(self.high) else np.asarray(self.high)
        
        if low.shape != (self.m,) or high.shape != (self.m,):
            raise ValueError("low и high должны быть скалярами или массивами длины m")
        if np.any(low >= high):
            raise ValueError("Каждое значение low должно быть строго меньше соответствующего high")
        
        samples = np.empty((self.n, self.m))
        for j in range(self.m):
            edges = np.linspace(low[j], high[j], self.n + 1)
            intervals = rng.uniform(edges[:-1], edges[1:])
            rng.shuffle(intervals)
            samples[:, j] = intervals
        
        self.features = samples  # сохраняем, как в random_array
        return self.features
    
    
    def generate_full_factorial(self, levels):
        """
        Генерирует полный факторный эксперимент: все возможные комбинации уровней факторов.
        
        Параметры:
            levels (int): 
                - если int: количество уровней для каждого из m факторов,
        
        Возвращает:
            np.ndarray: массив размером (N, m), где N = ∏ levels[i]
        """
        # Приводим low/high к массивам
        low = np.full(self.m, self.low) if np.isscalar(self.low) else np.asarray(self.low)
        high = np.full(self.m, self.high) if np.isscalar(self.high) else np.asarray(self.high)
        
        if low.shape != (self.m,) or high.shape != (self.m,):
            raise ValueError("low и high должны быть скалярами или массивами длины m")
        if np.any(low >= high):
            raise ValueError("Каждое значение low должно быть строго меньше соответствующего high")

        # Обрабатываем levels
        if np.isscalar(levels):
            levels_arr = np.full(self.m, levels, dtype=int)
        else:
            levels_arr = np.asarray(levels, dtype=int)
            if levels_arr.shape != (self.m,):
                raise ValueError("Параметр 'levels' должен быть int или массивом длины m")

        # Генерируем уровни для каждого фактора
        factor_levels = []
        for i in range(self.m):
            fl = np.linspace(low[i], high[i], num=levels_arr[i])
            factor_levels.append(fl)

        # Декартово произведение
        full_design = np.array(list(itertools.product(*factor_levels)))
        
        # Обновляем n и сохраняем
        self.n = full_design.shape[0]  # перезаписываем n
        self.features = full_design
        return self.features
        
    
    





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
    

class Functions():
    
    '''
    Функции для генерации датасетов
    '''
    
    def __init__(self):
        
        self.features = None
        self.target = None
        
        self.function = self.trigonometric
        self.function_name = 'trigonometric'
        self.random_seed = 42
        self.n_samples = 1000
        self.n_features = 3
        self.limits = (-10, 10)
        
        self.generate_random_array()
        

    
    
    # первая функция
    def ackley(self, x, y, z):
        term1 = -20 * np.exp(-0.2 * np.sqrt((x**2 + y**2 + z**2) / 3))
        term2 = -np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z)) / 3)
        function = term1 + term2 + 20 + np.e
        return function
    
    # вторая функция
    def exponential (self, x, y, z):
        function = np.exp(0.1*x) + np.exp(0.1*y) + np.exp(0.1*z) 
        return function
    
    
    # третья функция
    def gaussian (self,x, y):
        function = np.exp(-(x**2 + y**2)/2)
        return function
    
    
    # четвертая функция
    def hypersphere_4d (self, x1, x2, x3, x4):
        function = x1**2 + x2**2 + x3**2 + x4**2
        return function
    
    # пятая функция
    def hypersphere_5d (self, x1, x2, x3, x4, x5):
        function = x1**2 + x2**2 + x3**2 + x4**2 + x5**2
        return function
    
    # шестая функция
    def linear_5d (self, x1, x2, x3, x4, x5):
        function = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 1
        return function
    
    # восьмая функция
    def polynomial(self, x, y, z):
        function = 1 + 0.5 * (x**2 + y**2 + z**2) + 0.1 * (x**3 + y**3 + z**3)
        return function
    
    # девятая функция
    def product (self, x, y, z):
        function = x*y*z
        return function
        
    
    # десятая функция
    def trigonometric (self, x, y, z):
        function = np.sin(x)*np.cos(2*x) + np.sin(y)*np.cos(2*y) + np.sin(z)*np.cos(2*z)
        return function
    
   
    
    
    def set_function(self, function):
        
        self.function = function
    
        
        # определяем колиство признаков
        match self.function:
        
            case _ if self.function is self.ackley:
                self.n_features = 3
                self.function_name = 'ackley'
        
            case _ if self.function is self.exponential:
                self.n_features = 3
                self.function_name = 'exponential'
        
            case _ if self.function is self.gaussian:
               self.n_features = 2
               self.function_name = 'gaussian'    
            
            case _ if self.function is self.hypersphere_4d:
                self.n_features = 4
                self.function_name = 'hypersphere_4d'
            
            case _ if self.function is self.hypersphere_5d:
                self.n_features = 5
                self.function_name = 'hypersphere_5d'
                
            case _ if self.function is self.linear_5d:
                self.n_features = 5
                self.function_name = 'linear_5d'
            
            case _ if self.function is self.polynomial:
                self.n_features = 3
                self.function_name = 'polynomial'
        
            case _ if self.function is self.product:
                self.n_features = 3
                self.function_name = 'product'
        
            case _ if self.function is self.trigonometric:
                self.n_features = 3
                self.function_name = 'trigonometric'
                
        self.generate_random_array()
                
                 
    # функция для генерации массива данных для x, y, z
    def generate_random_array(self):
        """
        Генерирует массив размером (n, m) со случайными вещественными числами
        в диапазоне [low, high).
        
                
        Возвращает:
            np.ndarray: массив размером (n, m) типа float
        """
        
        seed = self.random_seed
        low = self.limits[0]
        high = self.limits[1]
        n = self.n_samples
        m = self.n_features
        
        # создаем генератор
        rng = np.random.default_rng(seed)
        
        # создаем матрицу признаков
        self.features = rng.uniform(low, high, size=(n, m))
        # Подставляем сгенерированный массив в функцию
        self.target = self.function (*self.features.T)
        
        
        
        
if __name__ == "__main__":
    
    # Исходные данные
    # создаем объект класса Functions 
    main_function = Functions()
    
    
    # определяем количетсво строк для обучения
    main_function.n_samples = 1000
    
    # пределы варьирования признаков
    main_function.limits = (-10, 10)
    
    # определяем вид функции
    main_function.set_function(main_function.trigonometric)
    
            
        
            
    data_generator = Generate_data(27, 3, -10, 10, seed = 42)
    hyper_cube = data_generator.generate_latin_hypercube()
    
    
