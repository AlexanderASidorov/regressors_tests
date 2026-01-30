# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error # методы оценки ошибки модели
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
import pyDOE3 as doe
from pyDOE3.doe_optimal import *




class Generate_coded_data():
    '''
    Класс для создания искуственных массивов признаков
    '''
    
    def __init__(self, function, n_samples, random_seed = None):
        '''
        Параметры:

        '''
        self.n_samples = n_samples
        self.function = function
        if random_seed is None:
            self.random_seed = self.function.random_seed
        else:
            self.random_seed = int(random_seed)
            

        
        
        self.matrix_types = ['ff2n', 'pbdesign', 'bbdesign', 'ccdesign', 'lhs', 'random_uniform', 'd_eff', 'pbdesign']
        self.matrix_type = 'lhs'
        
        
    def set_matrix_type (self, matrix_type, degree = 1):
        '''
        Определяем тип матрицы планирования
        Возможные варианты:
            ff2n - полный факторный эксперимент
            pbdesign - матрица Plackett-Burman
            bbdesign - матрица Box-Behnken
            ccdesign - матрица Central Composite
            lhs - матрица Latin-Hypercube
            random_uniform - случайные числа с нормальным распределением
            d_eff - D - оптимизированная матрица
            pbdesign - матрица Плакетта-Бермана
        '''
        
        if matrix_type not in self.matrix_types:
            raise ValueError (f'Можно выбрать только следующие типы матрицы планирвоания: {self.matrix_types}')
        else:
            self.matrix_type = matrix_type
        
        rng = np.random.default_rng(self.random_seed)
             
        # определяем матрицу планирования в кодированном масштабе
        match matrix_type:
            case 'ff2n':
                self.X_coded = doe.ff2n(self.function.n_features)
            case 'pbdesign':
                self.X_coded = doe.pbdesign(self.function.n_features)
            case 'bbdesign':
                self.X_coded = doe.bbdesign(self.function.n_features)
            case 'ccdesign':
                self.X_coded = doe.ccdesign(self.function.n_features)
            case 'lhs':
                
                X = doe.lhs(self.function.n_features, self.n_samples, seed = rng)
                self.X_coded = 2 * X - 1
            case 'random_uniform':
                self.X_coded = rng.uniform(-1., 1., 
                            size=(self.n_samples, self.function.n_features))
                
            case 'd_eff':
                n_levels=3
                n_factors=self.function.n_features
                            
                                
                
                X = generate_candidate_set(n_factors=n_factors, 
                                           n_levels=n_levels,
                                           grid_type = "full_factorial")
                j = 0
                step = 5
                n_points = int(self.n_samples - step)
                
                while j <= 5:
                    n_points += step
                    design, info = optimal_design(candidates = X,
                                                  n_points=n_points,
                                                  degree=degree,
                                                  criterion="D",
                                                  method="detmax")
                    
                    d_efficiency = info['D_eff']/100
                    j+=1
                    if d_efficiency >= 0.9:
                        break
                    
                self.X_coded = design
                
            case 'pbdesign':
               self.X_coded = doe.pbdesign(self.function.n_features)
                
                    
        
                
  
        
class Generate_natural_data():
    '''
    Класс для перехода от кодированного к натуральному масштабу и ообратно
    '''
    
    def __init__(self, function, X_coded):
        '''
        Параметры:
        '''
        self.function = function
        self.X_coded = X_coded
        
        self.target = None
        self.features = None
        
   
    def convert_coded_to_natural(self):
        """
        Преобразует матрицу плана из кодированного масштаба в натуральный,
        при условии, что ВСЕ факторы имеют одинаковые границы [low, high].
        
        """
        low, high = self.function.limits[0], self.function.limits[1]
        if low >= high:
            raise ValueError("low должно быть меньше high")
        
        # Линейное преобразование: [-1, +1] → [low, high]
        scale = (high - low) / 2.0
        center = (high + low) / 2.0
        
        # создаем матрицу признаков
        self.features = self.X_coded * scale + center
        
       
        # Подставляем сгенерированный массив в функцию
        self.target = self.function.main_function (*self.features.T)
        
        
    def convert_natural_to_coded(self):
        """
        Преобразует матрицу плана из натурального масштаба в кодированный,
        при условии, что ВСЕ факторы имеют одинаковые границы [low, high].
        
        """
        
        low, high = self.function.limits[0], self.function.limits[1]
        if low >= high:
            raise ValueError("low должно быть меньше high")
        
        
        scale = (high - low) / 2.0
        center = (high + low) / 2.0
        
        self.X_coded = (self.features - center) / scale
        


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
    r = np.corrcoef(y_true, y_pred)[0, 1] # коэффициент корреляции
    r2 = r2_score(y_true, y_pred) # коэффициент детерминации
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) #  среднеквадратическая ошибка
    mape = mean_absolute_percentage_error(y_true, y_pred) # cредняя абсолютная процентная ошибка
    # Строим график
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Ideal line')

    # Добавляем метрики на график
    metrics_text = f'R = {r:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAPE = {mape*100:.3f}%'
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
    Функции для определения базовой функции для дальней генерации 
    искусственного датасета
    '''
    
    def __init__(self):
        
                
        self.main_function = self.trigonometric
        self.main_function_name = 'trigonometric'
        self.random_seed = 1488
        self.n_samples = 1000
        self.n_features = 3
        self.limits = (-10, 10)
        
        #self.generate_random_array()
    
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
    def hyperbolic (self, x, y, z):
        function = np.tanh(x) + np.tanh(y) + np.tanh(z)
        return function
    
    # седьмая функция
    def linear_1d (self, x):
        function = 2*x + 1
        return function
    
    # восьмая функция
    def linear_2d (self, x, y):
        function = x + 2*y + 1
        return function
    
    # девятая функция
    def linear_4d (self, x1, x2, x3, x4):
        function = x1 + 2*x2 + 3*x3 + 4*x4 + 1
        return function
    
    
    # десятая функция
    def linear_5d (self, x1, x2, x3, x4, x5):
        function = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 1
        return function
    
    
    # одиннадцатая функция
    def logarithmic (self, x, y, z, c=11):
        function = np.log(abs(x) + c) + np.log(abs(y) + c) + np.log(abs(z) + c)  
        return function
    
    # двенадцатая функция
    def mixed_2D (self, x, y):
        function = x*y + np.sin(x) + np.cos(y)
        return function
      
    # тринадцатая функция
    def polynomial(self, x, y, z):
        function = 1 + 0.5 * (x**2 + y**2 + z**2) + 0.1 * (x**3 + y**3 + z**3)
        return function
    
    # четырнадцатая функция
    def product (self, x, y, z):
        function = x*y*z
        return function
    
    # пятнадцатая функция
    def quadratic (self, x, y, z):
        function = x**2 + y**2 + z**2
        return function
       
    # шестнадцатая функция
    def quadratic_1d (self, x):
        function = x**2 + 2*x + 1
        return function
    
    # семнадцатая функция
    def rastrigin (self, x, y, z, c=30, a = 10):
        term01 = x**2 - a*np.cos(2*np.pi*x)
        term02 = y**2 - a*np.cos(2*np.pi*y)
        term03 = z**2 - a*np.cos(2*np.pi*z)
        function = c + term01 + term02 + term03 
        return function
    
    # восемнадцатая функция
    def sin_1d (self, x):
        function = np.sin(2*np.pi*x)
        return function
    
    # девятнадцатая функция
    def sinusoidal (self, x, y, z):
        function = np.sin(x) + np.sin(y) + np.sin(z)
        return function
        
    
    # двадцатая функция
    def trigonometric (self, x, y, z):
        function = np.sin(x)*np.cos(2*x) + np.sin(y)*np.cos(2*y) + np.sin(z)*np.cos(2*z)
        return function
    
    
    
    # двадцать первая функция
    def rastrigin_10d(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        a = 10
        terms = [
            x1**2 - a * np.cos(2 * np.pi * x1),
            x2**2 - a * np.cos(2 * np.pi * x2),
            x3**2 - a * np.cos(2 * np.pi * x3),
            x4**2 - a * np.cos(2 * np.pi * x4),
            x5**2 - a * np.cos(2 * np.pi * x5),
            x6**2 - a * np.cos(2 * np.pi * x6),
            x7**2 - a * np.cos(2 * np.pi * x7),
            x8**2 - a * np.cos(2 * np.pi * x8),
            x9**2 - a * np.cos(2 * np.pi * x9),
            x10**2 - a * np.cos(2 * np.pi * x10),
        ]
        return 10 * 10 + sum(terms)
    
    # двадцать вторая функция
    def rosenbrock_8d(self, x1, x2, x3, x4, x5, x6, x7, x8):
        """
        8-мерная функция Розенброка.
        """
        coords = [x1, x2, x3, x4, x5, x6, x7, x8]
        total = 0.0
        for i in range(7):  # для 8 переменных есть 7 пар
            total += 100.0 * (coords[i+1] - coords[i]**2)**2 + (1.0 - coords[i])**2
        return total
    
    # двадцать вторая функция
    def ridge_demo_8d(self, x1, x2, x3, x4, x5, x6, x7, x8):
        """
       Демонстрационная функция для Ridge-регрессии (8 переменных).
        
        Содержит:
        - сильную мультиколлинеарность между x1, x2, x3 (почти одинаковые),
        - зависимость от их среднего: mean(x1,x2,x3),
        - квадратичный член: mean(x1,x2,x3)^2,
        - два взаимодействия: x4*x5 и x6*x7,
        - один шумовой признак x8 (не влияет на выход).
        
        При расширении до полинома 2-го порядка возникает сильная
        мультиколлинеарность между x1, x2, x3, x1^2, x1*x2, x2*x3 и т.д.,
        что делает Ridge-регрессию значительно эффективнее OLS.
        """
        # Среднее почти идентичных признаков
        m = (x1 + x2 + x3) / 3.0
        
        # Квадрат среднего
        term1 = 2.0 * m + 0.7 * (m ** 2)
        
        # Взаимодействия
        term2 = 1.1 * (x4 * x5)
        term3 = 0.9 * (x6 * x7)
        
        # x8 не участвует
        
        return term1 + term2 + term3
          
    
    def set_function(self, function):
        
        self.main_function = function
         
        func = self.main_function.__func__
    
        # определяем колиство признаков
        match func:
        
            case _ if func is Functions.ackley:
                self.n_features = 3
                self.main_function_name = 'ackley'
        
            case _ if func is Functions.exponential:
                self.n_features = 3
                self.main_function_name = 'exponential'
        
            case _ if func is Functions.gaussian:
               self.n_features = 2
               self.main_function_name = 'gaussian'    
            
            case _ if func is Functions.hypersphere_4d:
                self.n_features = 4
                self.main_function_name = 'hypersphere_4d'
            
            case _ if func is Functions.hypersphere_5d:
                self.n_features = 5
                self.main_function_name = 'hypersphere_5d'
                
            case _ if func is Functions.hyperbolic:
                self.n_features = 3
                self.main_function_name = 'hyperbolic' 
                
            case _ if func is Functions.linear_1d:
                self.n_features = 1
                self.main_function_name = 'linear_1d' 
                
            case _ if func is Functions.linear_2d:
                self.n_features = 2
                self.main_function_name = 'linear_2d'
                
            case _ if func is Functions.linear_4d:
                self.n_features = 4
                self.main_function_name = 'linear_4d'
                
                
            case _ if func is Functions.linear_5d:
                self.n_features = 5
                self.main_function_name = 'linear_5d'
            
            
            case _ if func is Functions.logarithmic:
                self.n_features = 3
                self.main_function_name = 'logarithmic'
                
            case _ if func is Functions.mixed_2D:
                self.n_features = 2
                self.main_function_name = 'mixed_2D'
            
    
            case _ if func is Functions.polynomial:
                self.n_features = 3
                self.main_function_name = 'polynomial'
        
            case _ if func is Functions.product:
                self.n_features = 3
                self.main_function_name = 'product'
                
            case _ if func is Functions.quadratic:
                self.n_features = 3
                self.main_function_name = 'quadratic'
                
            case _ if func is Functions.quadratic_1d:
                self.n_features = 1
                self.main_function_name = 'quadratic_1d'
                
            case _ if func is Functions.rastrigin:
                self.n_features = 3
                self.main_function_name = 'rastrigin'
                
            case _ if func is Functions.sin_1d:
                self.n_features = 1
                self.main_function_name = 'sin_1d'
                
            case _ if func is Functions.sinusoidal:
                self.n_features = 3
                self.main_function_name = 'sinusoidal'
        
            case _ if func is Functions.trigonometric:
                self.n_features = 3
                self.main_function_name = 'trigonometric'
                
            case _ if func is Functions.rastrigin_10d:
                self.n_features = 10
                self.main_function_name = 'rastrigin_10d'
                
            case _ if func is Functions.rosenbrock_8d:
                self.n_features = 8
                self.main_function_name = 'rosenbrock_8d'
                
            case _ if func is Functions.ridge_demo_8d:
                self.n_features = 8
                self.main_function_name = 'ridge_demo_8d'
                
                
            
                
                
    def get_function_dict(self):
        return {
            'ackley': self.ackley,
            'exponential': self.exponential,
            'gaussian': self.gaussian,
            'hypersphere_4d': self.hypersphere_4d,
            'hypersphere_5d': self.hypersphere_5d,
            'hyperbolic': self.hyperbolic,
            'linear_1d': self.linear_1d,
            'linear_2d': self.linear_2d,
            'linear_4d': self.linear_4d,
            'linear_5d': self.linear_5d,
            'logarithmic': self.logarithmic,
            'mixed_2D': self.mixed_2D,
            'polynomial': self.polynomial,
            'product': self.product,
            'quadratic': self.quadratic,
            'quadratic_1d': self.quadratic_1d,
            'rastrigin': self.rastrigin,
            'sin_1d': self.sin_1d,
            'sinusoidal': self.sinusoidal,
            'trigonometric': self.trigonometric,
            'rastrigin_10d': self.rastrigin_10d,
            'rosenbrock_8d': self.rosenbrock_8d,
            'ridge_demo_8d': self.ridge_demo_8d
        }
                

    
class Matrix_extension ():
    def __init__(self, X_coded):
        
        self.X_coded = X_coded
        self.n_features = X_coded.shape[1]
        self.basis_functions = None 
        self.set_basis_functions('safe')
        self.X_coded_ext = None
    
    
    def set_basis_functions(self, basis_type, include_interactions=True):
        """Выбор набора базисных функций в зависимости от типа"""
        
        self.basis_type = basis_type
        self.include_interactions = include_interactions
        
        if basis_type == 'safe':
            # Минимальный безопасный набор
            self.basis_functions = [
                lambda x: x**2,
                lambda x: x**3,
                lambda x: np.sin(np.pi * x),
                lambda x: np.cos(np.pi * x),
                lambda x: np.tanh(x),
                lambda x: np.exp(-x**2),
            ]
            
        elif basis_type == 'chebyshev':
            # Полиномы Чебышева (рекомендуется для [-1, 1])
            self.basis_functions = [
                lambda x: x,                    # T1
                lambda x: 2*x**2 - 1,           # T2
                lambda x: 4*x**3 - 3*x,         # T3
                lambda x: 8*x**4 - 8*x**2 + 1,  # T4
                lambda x: 16*x**5 - 20*x**3 + 5*x,  # T5
            ]
            
        elif basis_type == 'fourier':
            # Тригонометрический базис
            self.basis_functions = [
                lambda x: np.sin(np.pi * x),
                lambda x: np.cos(np.pi * x),
                lambda x: np.sin(2*np.pi * x),
                lambda x: np.cos(2*np.pi * x),
                lambda x: np.sin(3*np.pi * x),
                lambda x: np.cos(3*np.pi * x),
            ]
            
        elif basis_type == 'full':
            # Полный гибридный (с осторожностью)
            self.basis_functions = [
                # Чебышев
                lambda x: 2*x**2 - 1,
                lambda x: 4*x**3 - 3*x,
                
                # Фурье
                lambda x: np.sin(np.pi * x),
                lambda x: np.cos(np.pi * x),
                lambda x: np.sin(2*np.pi * x),
                
                # Гауссовы
                lambda x: np.exp(-x**2),
                lambda x: np.exp(-4*x**2),  # ужече
                lambda x: x * np.exp(-x**2),  # нечетная
                
                # Сигмоиды
                lambda x: np.tanh(2*x),     # круче
                lambda x: x / (1 + np.abs(x)),
            ]
            
            
        elif basis_type == 'custom':
        # Набор из функции build_custom_features
            self.basis_functions = [
                lambda x: x**2,                       # квадрат
                lambda x: np.sin(x),                  # sin(x)
                lambda x: np.cos(x),                  # cos(x)
                lambda x: np.exp(np.clip(x, -5, 5)),  # exp(x) с защитой
                lambda x: np.tanh(x),                 # tanh(x)
                lambda x: np.log(np.abs(x) + 1e-6),   # log(|x| + eps)
            ]
            
        elif basis_type == 'no_extension':
            pass
        else:
            raise ValueError(f"Unknown basis_type: {basis_type}")
     
            
    def expand_features(self):
        '''
        Расширяем пространство
        '''
        X = np.asarray(self.X_coded)
        n_samples = X.shape[0]
        
        
        if self.basis_type == 'no_extension':
            
            blocks = [
                        np.ones((n_samples, 1)),  # столбец единиц
                        X                         # исходные признаки
                        ]
            self.X_coded_ext = np.hstack(blocks)
        
        # исли признаков больше 5, то запрещаем расширение матрицы признаков
        elif self.n_features >= 5:
            self.basis_type = 'no_extension'
            blocks = [
                        np.ones((n_samples, 1)),  # столбец единиц
                        X                         # исходные признаки
                        ]
            self.X_coded_ext = np.hstack(blocks)
            
        
            
        
        else:
                      
            # Создаем список блоков
            blocks = [
                np.ones((n_samples, 1)),  # столбец единиц
                X                         # исходные признаки
            ]
            
            # Базисные функции
            for func in self.basis_functions:
                X_new = func(X)
                X_new = np.nan_to_num(X_new, nan=0.0, posinf=1.0, neginf=-1.0)
                blocks.append(X_new)
            
            # Взаимодействия
            if self.include_interactions and X.shape[1] >= 2:
                for i in range(X.shape[1]):
                    for j in range(i+1, X.shape[1]):
                        blocks.append(X[:, [i]] * X[:, [j]])  # проще со срезами
            
            self.X_coded_ext = np.hstack(blocks)
        
        


def run_experiment(
                    n_samples=200,
                    function_name='hyperbolic',
                    matrix_type='lhs',
                    basis_functions_set='full',
                    random_seed=1488,
                    test_ratio=0.25,
                    use_lasso=False,
                    lasso_cv=5,
                    plot_title_prefix='Тестирование поверхности отклика',
                    print_recap = True
                ):
    """
    Запускает полный цикл генерации данных, обучения модели и визуализации результата.

    Параметры:
        n_samples (int): Количество обучающих точек.
        function_name (str): Название функции из доступного списка.
        matrix_type (str): Тип матрицы планирования ('ff2n', 'pbdesign', 'bbdesign', 'ccdesign', 'lhs', 'random_uniform').
        basis_functions_set (str): Набор базисных функций ('safe', 'chebyshev', 'fourier', 'full', 'custom').
        random_seed (int): Фиксированный seed для воспроизводимости.
        test_ratio (float): Доля тестовых данных относительно обучающих (например, 0.25 → 25%).
        use_lasso (bool): Использовать LassoCV вместо LinearRegression.
        lasso_cv (int): Число фолдов для кросс-валидации в LassoCV.
        plot_title_prefix (str): Префикс заголовка графика.

    Возвращает:
        model: обученная модель.
        r2: коэффициент детерминации на тесте.
    """
   

    # --- Инициализация функции ---
    main_function = Functions()
    main_function.random_seed = random_seed
    functions = main_function.get_function_dict()
    if function_name not in functions:
        raise ValueError(f"Неизвестная функция: {function_name}. Доступные: {list(functions.keys())}")
    main_function.set_function(functions[function_name])

    n_test_samples = int(n_samples * test_ratio)

    # --- Обучающая выборка ---
    training_matrix = Generate_coded_data(main_function, n_samples, random_seed=random_seed)
    training_matrix.set_matrix_type(matrix_type)
    training_X_coded = training_matrix.X_coded

    training_data = Generate_natural_data(main_function, training_X_coded)
    training_data.convert_coded_to_natural()
    training_features = training_data.features
    training_target = training_data.target

    # --- Расширение признаков ---
    basis_functions = Matrix_extension(training_X_coded)
    basis_functions.set_basis_functions(basis_functions_set)
    basis_functions.expand_features()
    X_coded_ext_training = basis_functions.X_coded_ext

    # --- Обучение модели ---
    if use_lasso:
        model = LassoCV(cv=lasso_cv, random_state=random_seed, max_iter=10000)
    else:
        model = LinearRegression(fit_intercept=False)
    model.fit(X_coded_ext_training, training_target)

    # --- Тестовая выборка ---
    testing_matrix = Generate_coded_data(main_function, n_test_samples, random_seed=random_seed)
    testing_matrix.set_matrix_type('random_uniform')
    testing_X_coded = testing_matrix.X_coded

    testing_data = Generate_natural_data(main_function, testing_X_coded)
    testing_data.convert_coded_to_natural()
    testing_features = testing_data.features
    testing_target = testing_data.target

    testing_basis_functions = Matrix_extension(testing_X_coded)
    testing_basis_functions.set_basis_functions(basis_functions_set)
    testing_basis_functions.expand_features()
    X_coded_ext_testing = testing_basis_functions.X_coded_ext

    testing_target_model = model.predict(X_coded_ext_testing)

    # --- Визуализация и метрика ---
    plot_true_vs_predicted(
        testing_target,
        testing_target_model,
        title=f"{plot_title_prefix}. Функция {main_function.main_function_name}"
    )

    r2 = r2_score(testing_target, testing_target_model)
    
    if print_recap:
        print (f'Тип функции: {function_name}, Количество экспериментов: {n_samples}, R² = {r2:.4f}')
    
    initial_data = {'function_name': function_name,
                    'training_X_coded': training_X_coded, 
                    'training_features':training_features, 
                    'training_target': training_target,
                    'X_coded_ext_training': X_coded_ext_training,
                    'testing_X_coded': testing_X_coded,
                    'testing_features': testing_features,
                    'testing_target': testing_target,
                    'X_coded_ext_testing': X_coded_ext_testing
                    }
    
    return model, r2, initial_data  




        
 #%%       
        
if __name__ == "__main__":
    
    # Тип функции нужно выбрать из следующего списка:
    """
     ackley', 'exponential','gaussian','hypersphere_4d','hypersphere_5d','hyperbolic',
    'linear_1d','linear_2d','linear_4d','linear_5d', 'logarithmic','mixed_2D',
    'polynomial','product','quadratic','quadratic_1d','rastrigin','sin_1d',
    'sinusoidal', 'trigonometric', 'rastrigin_10d'
    """
    
    
    names_of_functions =  ['ackley', 'exponential','gaussian','hypersphere_4d','hypersphere_5d','hyperbolic',
                           'linear_1d','linear_2d','linear_4d','linear_5d', 'logarithmic','mixed_2D',
                           'polynomial','product','quadratic','quadratic_1d','rastrigin','sin_1d',
                           'sinusoidal', 'trigonometric', 'rastrigin_10d', 'rosenbrock_8d', 'ridge_demo_8d']
    n_samples = 52
    problematic_models = {}
    
    i = 0
    
    for item in names_of_functions:
        model, r2, initial_data = run_experiment(
            n_samples= n_samples,
            function_name= item,
            matrix_type='lhs',              
            basis_functions_set='full', # возможные варианты: safe, chebyshev, fourier, full, custom, no_extension
            random_seed=1488
        )
        
        if r2<0.8:
            i+=1
            problematic_models[item] = r2
            
    print('################')
    print('################')
    print (f'Проблемные модели при {n_samples} экспериментах ({i} моделей) :') 
    print (problematic_models)
    
    
 #%%   
    model, r2, initial_data = run_experiment(n_samples= n_samples,
                                             function_name= 'rosenbrock_8d',
                                             matrix_type='pbdesign',              
                                             basis_functions_set='no_extension', # возможные варианты: safe, chebyshev, fourier, full, custom, no_extension
                                             random_seed=1488
                                             )
    
        
        
#%%%

    # Пример: L9(3^4) — 9 экспериментов, 4 фактора, 3 уровня
    oa = doe.get_orthogonal_array("L8(2^7)")  # возвращает массив с уровнями 1, 2, 3

    # Преобразуем в 0, 1, 2 (если нужно для моделирования)
    oa_zero_based = oa - 1

    
    
    
    

    
