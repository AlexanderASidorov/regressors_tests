# Neural Network Model Exported from C++
# Generated on: 2025-12-03T15:33:07
# Input dimensions: 3
# Output dimensions: 1
# Training samples: 1000
# Architecture: 3-24-1, LR: 0.001, Epochs: 10000

import numpy as np

"""
Improved Neural Network Model:
  - Input layer: 3 neurons
  - Hidden layer: 24 neurons (swish activation)
  - Output layer: 1 neurons (linear activation)
  - Features: data normalization, adaptive architecture
"""

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

def swish(x):
    return x * sigmoid(x)

class DataNormalizer:
    def __init__(self):
        self.input_mean = np.array([0.00956376447656674, 0.0338122988232822, 0.155905884388757])
        self.input_std = np.array([6.15810031569971, 5.99560034302561, 6.12577798064118])
        self.output_mean = np.array([3.61265787])
        self.output_std = np.array([1.24887737665464])

    def transform_input(self, x):
        return (np.array(x) - self.input_mean) / self.input_std

    def inverse_transform_output(self, y):
        return np.array(y) * self.output_std + self.output_mean

class NNModel:
    def __init__(self):
        self.input_dim = 3
        self.output_count = 1
        self.normalizer = DataNormalizer()

        # Weights and biases from C++ training
        self.weights_input_hidden = np.array([
            [-0.0231884224516903, -0.576081791945968, 1.35366435555688, -0.00930483993471719, 0.00905367864785624, 0.0248865085391756, 0.00921206350060871, 0.00895795691833125, 0.00906202713670119, 0.00760287160378402, 0.00499404042904273, 0.0231403069182876, 0.0146038315876755, 0.00147779004667618, 0.00949791680233821, 0.00935398129121278, 0.192141144746152, -0.0144359663866531, 0.00688058704084395, -0.0180591028608659, 0.104807125394208, 0.015413196552102, 0.00941123671447731, -0.0198024324311006],
            [-0.0089710349442048, -0.564633943143377, 0.115406927017701, 0.0165876836162012, -0.00178674672127233, 0.0118372171299314, -0.00202292760461065, -0.00112636655422721, 0.0150557358392837, 0.000610078103463619, 0.00233038047288738, 0.0182545452332206, 0.00869218848480156, 0.00110478830020893, -0.00225720450019782, 0.0146983200061798, 0.159464325872222, -0.0103834251465269, 0.00120290355267896, -0.00861324603083884, 1.3131949279841, 0.00739157826276572, -0.00194957383876954, -0.0328854741027039],
            [0.00601006045273955, -0.574507568842315, 0.09825690117282, 0.0156654979688344, -0.00635170118340125, -0.00401681663629314, -0.00646461859995244, -0.00656550721948755, 0.0105505172993473, -0.00566122659785486, -0.00336068312908193, 0.000225859064135969, 0.000515822453796032, -0.000184665241567332, -0.00672004044865536, 0.00996543131506391, 1.32318438467325, 0.000406166373259817, -0.00518797997767374, 0.00473882773772408, 0.102155963197496, -0.0011990576821481, -0.00673204135984548, -0.00143849783378697],
        ])

        self.biases_hidden = np.array([0.00343597406498353, 0.326404614566573, -0.486113349943454, 0.00133225415593023, -0.00438836600223461, 0.00651860097100695, -0.00417572494480074, -0.00564005084973366, -0.0035009000309887, -0.00676550480630408, -0.00629831096971362, 0.00784674035936196, -0.000144668116968851, -0.00149793922415044, -0.00412728442608856, -0.00314238692859779, -0.444527540272073, 0.00291065122373047, -0.00693753848532959, 0.00816705165057245, -0.522273527384664, -0.000419713099106992, -0.00460840500484209, 0.0282550133847959])

        self.weights_hidden_output = np.array([
            [0.00778032121082515],
            [-0.729060414024856],
            [0.927128787081477],
            [-0.00546206331549997],
            [-0.00140890824963209],
            [-0.00634467875280478],
            [-0.00130054519163238],
            [-0.00186864313389672],
            [-0.00663746097891141],
            [-0.00255901319940422],
            [-0.00280803665554508],
            [-0.008020982546725],
            [-0.00501823996315774],
            [-0.000934743841677315],
            [-0.0012382701095851],
            [-0.00652126965267072],
            [0.936211244366749],
            [0.00686345099585546],
            [-0.00271603578523054],
            [0.00769446755262294],
            [0.858028826853647],
            [-0.00477427817658187],
            [-0.001430245055412],
            [0.0223942044866403],
        ])

        self.biases_output = np.array([-0.0424982059941574])

    def predict(self, inputs):
        """
        Predict output for given input vector
        
        Args:
            inputs: list of input values (length must match input_dim)
            
        Returns:
            list of output values
        """
        inputs = np.array(inputs, dtype=float)
        
        if inputs.shape != (self.input_dim,):
            raise ValueError(f'Input must be a vector of size {self.input_dim}. Got shape {inputs.shape}')
        
        # Normalize input
        normalized_input = self.normalizer.transform_input(inputs)
        
        # Forward pass through the network
        # Input -> Hidden layer (swish activation)
        hidden_inputs = np.dot(normalized_input, self.weights_input_hidden) + self.biases_hidden
        hidden_outputs = swish(hidden_inputs)
        
        # Hidden -> Output layer (linear activation)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.biases_output
        
        # Denormalize output
        final_outputs = self.normalizer.inverse_transform_output(final_inputs)
        
        return final_outputs.tolist()

    def get_architecture(self):
        """Return network architecture information"""
        return {
            'input_dim': self.input_dim,
            'hidden_units': 24,
            'output_count': self.output_count,
            'total_parameters': 121
        }

# Example usage:
if __name__ == '__main__':
    model = NNModel()
    test_input = [0.0, 0.0, 0.0]
    output = model.predict(test_input)
    print(f'Input: {test_input}')
    print(f'Output: {output}')
    
    
    
        
    

    from sklearn.metrics import mean_squared_error, r2_score
    
    # импортируем функции
    from functions import generate_random_array, plot_true_vs_predicted
    
    
    
    # вторая функция
    def exponential (x, y, z):
        function = np.exp(0.1*x) + np.exp(0.1*y) + np.exp(0.1*z) 
        return function
    
    # вид функции
    main_function = exponential
    
    # назначаем количество строк в генерируемом датасете 
    n_samples = 1000
    
    n_features = 3
    
    
    # пределы варьирования признаков
    limits = (-10, 10)


    
    
    
    
    # генерим случайный массив признаков от -10 до +10
    features = generate_random_array(n_samples, n_features, limits[0], limits[1], seed = 1488)
    # Подставляем сгенерированный массив в функцию
    target = main_function (*features.T)
    
    results = np.zeros((len(target), 5))
    
    
    for i in range (len(target)):
        results[i, :3] = features[i,:]
        results[i, 3] = model.predict(results[i, :3])[0]
        results[i, 4] = target[i]
        
    r2 = r2_score(results[:, 4], results[:, 3])
        
    
    
    
