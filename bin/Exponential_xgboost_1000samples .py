# XGBoost Model Exported from C++
# Generated on: 2025-12-03T15:37:29
# Input dimensions: 3
# Output dimensions: 1
# Training samples: 1000
# Pure Python XGBoost Implementation - Auto-tune: ON

import numpy as np
import math








"""
Pure Python XGBoost Implementation
This model was trained in C++ and exported to Python.
It contains the complete tree structures for fast inference.

Model Architecture:
  - Gradient Boosting with decision trees
  - Multi-output support (each output has its own ensemble)
  - No external dependencies required
"""

def create_tree_from_dict(tree_dict):
    if tree_dict is None:
        return None
    
    node = TreeNode()
    node.feature = tree_dict['feature']
    node.threshold = tree_dict['threshold']
    node.value = tree_dict['value']
    
    if 'left' in tree_dict:
        node.left = create_tree_from_dict(tree_dict['left'])
    if 'right' in tree_dict:
        node.right = create_tree_from_dict(tree_dict['right'])
    
    return node

class TreeNode:
    def __init__(self, feature=-1, threshold=0.0, value=0.0, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, x):
        if self.is_leaf():
            return self.value
        
        if self.feature < 0 or self.feature >= len(x):
            return self.value
        
        if x[self.feature] < self.threshold:
            if self.left:
                return self.left.predict(x)
        else:
            if self.right:
                return self.right.predict(x)
        return self.value

class XGBTree:
    def __init__(self, tree_dict=None, learning_rate=0.1):
        self.learning_rate = learning_rate
        if tree_dict:
            self.root = create_tree_from_dict(tree_dict)
        else:
            self.root = None

    def predict(self, x):
        if self.root is None:
            return 0.0
        return self.learning_rate * self.root.predict(x)

class XGBoostModel:
    def __init__(self):
        self.input_dim = 3
        self.output_count = 1
        self.biases = []
        self.trees = []
        self._initialize_model()

    def _initialize_model(self):
        totalTrees = 0
        
        # Output 0
        trees_output_0 = []
        self.biases.append(3.56992177)
        
        tree_dict_0_0 = {
            'feature': 1,
            'threshold': 2.88455864672239,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 2.44981604234887,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -0.89098170965654,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.02988995770383,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.405013956861023,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 7.26043004474951,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0161944188055939,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.677506682665025,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 6.59312067374141,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': 5.47843839803558,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0960895085337135,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.808554581207323,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.3337539148287,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.60648610699002,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.18984079435397,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_0 = XGBTree(tree_dict_0_0, learning_rate=0.05)
        trees_output_0.append(tree_0_0)
        totalTrees += 1
        tree_dict_0_1 = {
            'feature': 1,
            'threshold': 0.870005138647091,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 5.08824194572204,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -3.77221677781244,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.13805225876167,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.506854093351114,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': -8.33021613484446,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0504971663065284,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.624334090476051,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 2.28692184508014,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': 7.11609203669763,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0613762350248268,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.725736689571741,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.76364586390438,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.15963017568147,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.15148317130906,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_1 = XGBTree(tree_dict_0_1, learning_rate=0.05)
        trees_output_0.append(tree_0_1)
        totalTrees += 1
        tree_dict_0_2 = {
            'feature': 0,
            'threshold': 3.66702030830034,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 3.69140665896816,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -1.89334092313824,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.07293635528326,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.508378681186077,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 7.81838490544263,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0133676999681209,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.786764451470364,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 5.07902830414415,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 6.10484780628176,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0966052806421661,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.827836143702668,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 8.98151123829985,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.61660062047774,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.74720788149808,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_2 = XGBTree(tree_dict_0_2, learning_rate=0.05)
        trees_output_0.append(tree_0_2)
        totalTrees += 1
        tree_dict_0_3 = {
            'feature': 0,
            'threshold': 2.56727103208237,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -1.07574414908667,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -2.19725421420375,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.22046846010331,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.745469227023502,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 5.60510990120615,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.246976331178879,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.549174721912532,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.82167140218961,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 6.25601582677108,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0057615799430513,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.716876759158655,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.91882911190069,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.39196026736124,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.31447751618617,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_3 = XGBTree(tree_dict_0_3, learning_rate=0.05)
        trees_output_0.append(tree_0_3)
        totalTrees += 1
        tree_dict_0_4 = {
            'feature': 2,
            'threshold': 4.6882221909939,
            'value': 0,
            'left': {
                'feature': 0,
                'threshold': 2.2341122499746,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 0.524904220109017,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.00046284115386,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.194906718704498,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 6.1350803297051,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0191651348480654,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.677062222171969,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 0,
                'threshold': 0.273250023290496,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.84438879139005,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.161577891091494,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.849494064718273,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 9.03738323064584,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.46080911494852,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.40069995538137,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_4 = XGBTree(tree_dict_0_4, learning_rate=0.05)
        trees_output_0.append(tree_0_4)
        totalTrees += 1
        tree_dict_0_5 = {
            'feature': 2,
            'threshold': 2.39157824637521,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 3.52906310890248,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -3.96516191763301,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.0714477112075,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.518111019709333,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.51222419830909,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0820896490727559,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.805949438669393,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 4.53935886317331,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.21912255607886,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.245292929628174,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.795898613581308,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.93597329545305,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.47895540755372,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.37076836321523,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_5 = XGBTree(tree_dict_0_5, learning_rate=0.05)
        trees_output_0.append(tree_0_5)
        totalTrees += 1
        tree_dict_0_6 = {
            'feature': 2,
            'threshold': 2.39157824637521,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -0.89098170965654,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -2.20194259814481,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -1.13873609416622,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.605722891876864,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.11609203669763,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.151973605912752,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.576679127355634,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.90327453742386,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.63610641743393,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.186175396490755,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.04162029456449,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.45766705094171,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.40759993180882,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.2830715362751,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_6 = XGBTree(tree_dict_0_6, learning_rate=0.05)
        trees_output_0.append(tree_0_6)
        totalTrees += 1
        tree_dict_0_7 = {
            'feature': 0,
            'threshold': 4.81202742062617,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 3.75735853211681,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -3.0024461469222,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.911855774760275,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.384444757998394,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 0.273250023290496,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.208876125466896,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.87981691202868,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': -0.571174112567409,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.76644143939207,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0866497440993478,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.663407723994085,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 6.89788851016854,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.14072317668103,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.8450677320459,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_7 = XGBTree(tree_dict_0_7, learning_rate=0.05)
        trees_output_0.append(tree_0_7)
        totalTrees += 1
        tree_dict_0_8 = {
            'feature': 1,
            'threshold': 1.31776690162502,
            'value': 0,
            'left': {
                'feature': 0,
                'threshold': 2.13301335870507,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -1.33294013663875,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.983890920866805,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.479689146388037,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 6.25601582677108,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.218241696736821,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.522351922373485,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 0,
                'threshold': 3.74417468233127,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': 7.56482648867538,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.00319850951700996,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.681324095124518,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.39514444340635,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.897436277204793,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.54946588078335,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_8 = XGBTree(tree_dict_0_8, learning_rate=0.05)
        trees_output_0.append(tree_0_8)
        totalTrees += 1
        tree_dict_0_9 = {
            'feature': 2,
            'threshold': 4.71130375941182,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 0.913161994582515,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 0.0016004152278617,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.846692649989573,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.240195650093741,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.56482648867538,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.00846743404480742,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.685128743536854,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.03423646428921,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -2.65578143132426,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.209603099123117,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.642695934751665,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.93597329545305,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.30531471584023,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.08522132754881,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_9 = XGBTree(tree_dict_0_9, learning_rate=0.05)
        trees_output_0.append(tree_0_9)
        totalTrees += 1
        tree_dict_0_10 = {
            'feature': 1,
            'threshold': 3.82064037944921,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 2.28528166394293,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -2.20480546973504,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.892347823797962,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.38581141706902,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 6.55244679453874,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.102118322564208,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.313733149451347,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 2.33053360299731,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': 7.11609203669763,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0194603368682544,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.718633034903023,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.93597329545305,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.10598476107666,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.93342177728873,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_10 = XGBTree(tree_dict_0_10, learning_rate=0.05)
        trees_output_0.append(tree_0_10)
        totalTrees += 1
        tree_dict_0_11 = {
            'feature': 0,
            'threshold': 2.47275386031179,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 3.22088489766914,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -2.09847609371074,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.875085283709039,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.390185335754156,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.5703590147733,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0310212026620611,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.613607398551409,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 0.870005138647091,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 6.22578330334774,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.117655811426363,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.531347253078048,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 9.7586528589638,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.05373185209664,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.97432947720861,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_11 = XGBTree(tree_dict_0_11, learning_rate=0.05)
        trees_output_0.append(tree_0_11)
        totalTrees += 1
        tree_dict_0_12 = {
            'feature': 0,
            'threshold': 2.56727103208237,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 2.31761083177883,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -3.54548259376118,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.837484337873161,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.382327720176221,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.5703590147733,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.058250937328985,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.470816342658173,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.6943509590247,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.13936947358214,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.11440758853786,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.762773761303128,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 9.23076923076923,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.942880582876508,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 2.01313602856994,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_12 = XGBTree(tree_dict_0_12, learning_rate=0.05)
        trees_output_0.append(tree_0_12)
        totalTrees += 1
        tree_dict_0_13 = {
            'feature': 0,
            'threshold': -0.293243271584166,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 0.103966100309316,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -4.08986876558968,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.989690949684813,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.631452751519886,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 5.68394122662662,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.179346488936256,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.276967962191225,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 2.36090538367564,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.04107487785346,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.104406559778746,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.6285253594066,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.32603936171002,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.790440920677129,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.35020621880656,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_13 = XGBTree(tree_dict_0_13, learning_rate=0.05)
        trees_output_0.append(tree_0_13)
        totalTrees += 1
        tree_dict_0_14 = {
            'feature': 2,
            'threshold': 4.60756738655739,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 4.27106345764573,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -2.35270402589618,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.794477312390186,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.227460291614593,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.45284321969994,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.129517097883153,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.687310817900083,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.49644342286868,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -5.34316256882355,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.104521007474412,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.661973129444345,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.61771059100195,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.13836453688751,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.85524864208923,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_14 = XGBTree(tree_dict_0_14, learning_rate=0.05)
        trees_output_0.append(tree_0_14)
        totalTrees += 1
        tree_dict_0_15 = {
            'feature': 2,
            'threshold': 1.1028214540842,
            'value': 0,
            'left': {
                'feature': 0,
                'threshold': 2.47275386031179,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.486325670509,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.851827680677524,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.513449397164173,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.86021616487757,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0343483050565445,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.662948723756031,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 0,
                'threshold': 3.76376790147335,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.41890123393038,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0897477059871814,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.579791991290801,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 5.07837117888472,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.815733636092162,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.28639899721009,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_15 = XGBTree(tree_dict_0_15, learning_rate=0.05)
        trees_output_0.append(tree_0_15)
        totalTrees += 1
        tree_dict_0_16 = {
            'feature': 2,
            'threshold': -1.23495128919726,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 4.27295290287362,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -3.07092128799532,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.885285134445493,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.431323451196359,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.47356121511482,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0199886329424913,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.488195852551556,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 2.09955567211766,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 7.17651995383188,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0666832133776117,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.377176554726187,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 7.86771287216719,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.508805099691486,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.12509753336909,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_16 = XGBTree(tree_dict_0_16, learning_rate=0.05)
        trees_output_0.append(tree_0_16)
        totalTrees += 1
        tree_dict_0_17 = {
            'feature': 0,
            'threshold': 1.79564363778407,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 0.103966100309316,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -1.6276991756014,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.792136782757421,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.44200574540539,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 5.60510990120615,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.19245565188564,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.27990336021689,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.90327453742386,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 6.67413667476692,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0440776554829657,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.533809195885004,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.39514444340635,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.746129525306385,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.20150937270719,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_17 = XGBTree(tree_dict_0_17, learning_rate=0.05)
        trees_output_0.append(tree_0_17)
        totalTrees += 1
        tree_dict_0_18 = {
            'feature': 2,
            'threshold': 5.03836409882358,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 4.27106345764573,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -1.77222600894416,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.621455229500836,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.176722704403185,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 0.141626963045008,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.10313522643689,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.613063069983492,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 2.08083474155924,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.63610641743393,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.100964534771186,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.686461429253669,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.93597329545305,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.877493015106503,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.6771139552121,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_18 = XGBTree(tree_dict_0_18, learning_rate=0.05)
        trees_output_0.append(tree_0_18)
        totalTrees += 1
        tree_dict_0_19 = {
            'feature': 2,
            'threshold': 5.03836409882358,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 2.32370489431367,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.2345047249925,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.651817581238682,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.229908955987711,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.27124562704276,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0307600919225258,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.409644048059321,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 4.10277098313046,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.63610641743393,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.211376773806453,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.799489063212103,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.86970031469523,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.890014378111891,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.54366451692138,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_19 = XGBTree(tree_dict_0_19, learning_rate=0.05)
        trees_output_0.append(tree_0_19)
        totalTrees += 1
        tree_dict_0_20 = {
            'feature': 2,
            'threshold': 0.309028949378511,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 6.16170157255703,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -2.44901534380783,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.656844800532537,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.284832320073022,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': -8.48046034684537,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.3762113241356,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.337012371871498,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 0.975369351237912,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 7.26043004474951,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0611420616776913,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.345893084928706,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.72136442941913,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.503075388221718,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.939405964433417,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_20 = XGBTree(tree_dict_0_20, learning_rate=0.05)
        trees_output_0.append(tree_0_20)
        totalTrees += 1
        tree_dict_0_21 = {
            'feature': 0,
            'threshold': 6.60726775490418,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': -1.64818163151288,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -1.60488973087631,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.735188592555509,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.271874177587964,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.15093297662476,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0754931263865964,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.520574038199533,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 0.481299712964097,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.92520465322592,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.215970479462207,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.699677796078068,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.38727117112031,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.961148476619413,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.76663684595837,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_21 = XGBTree(tree_dict_0_21, learning_rate=0.05)
        trees_output_0.append(tree_0_21)
        totalTrees += 1
        tree_dict_0_22 = {
            'feature': 0,
            'threshold': 3.08098728786026,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 1.04449316282572,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.486325670509,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.714515031394632,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.360078535851394,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.98785226756348,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.00763899156543799,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.522705436635633,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 0.931949571448229,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.0886104273484,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.128472641346225,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.517323620395413,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 6.8956570646626,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.552964921200332,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.09531607376854,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_22 = XGBTree(tree_dict_0_22, learning_rate=0.05)
        trees_output_0.append(tree_0_22)
        totalTrees += 1
        tree_dict_0_23 = {
            'feature': 2,
            'threshold': -0.571174112567409,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 1.60137313351833,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.08597664155661,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.650946053043454,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.311823025700002,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.11609203669763,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.130662072114898,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.25710322197986,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 2.36472984522731,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 7.71025827803923,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0842705900532172,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.41233626523472,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 6.60225169614777,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.531238529854127,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.01998711719343,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_23 = XGBTree(tree_dict_0_23, learning_rate=0.05)
        trees_output_0.append(tree_0_23)
        totalTrees += 1
        tree_dict_0_24 = {
            'feature': 0,
            'threshold': 4.51802969971573,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -3.54548259376118,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.76032269772678,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.740189338845489,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.420823956979566,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 5.36170212846611,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.182221501610896,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.182904747015463,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.39514444340635,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.87933808535418,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.247704300421468,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.662357776111466,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.070725690709,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.905554156953394,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.31336426204583,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_24 = XGBTree(tree_dict_0_24, learning_rate=0.05)
        trees_output_0.append(tree_0_24)
        totalTrees += 1
        tree_dict_0_25 = {
            'feature': 0,
            'threshold': 2.56727103208237,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 5.60510990120615,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -4.0484328100585,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.572232417025565,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.237625592585924,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -8.26236628243959,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.126917467488334,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.231077735028246,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 0.931949571448229,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.73486763891885,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.121892387524959,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.662116983677818,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 6.83616498943853,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.471644010307614,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.951841595641671,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_25 = XGBTree(tree_dict_0_25, learning_rate=0.05)
        trees_output_0.append(tree_0_25)
        totalTrees += 1
        tree_dict_0_26 = {
            'feature': 2,
            'threshold': 2.28528166394293,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 1.44666613895684,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.4320445153754,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.642768364733761,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.309346024695918,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 7.11609203669763,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.100132302010156,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.244312005249526,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': -2.65578143132426,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.63610641743393,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0676672979293512,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.468667140916285,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.22962232300812,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.389879330683847,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.869019151224001,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_26 = XGBTree(tree_dict_0_26, learning_rate=0.05)
        trees_output_0.append(tree_0_26)
        totalTrees += 1
        tree_dict_0_27 = {
            'feature': 0,
            'threshold': 3.66702030830034,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 4.71130375941182,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -3.99617088754663,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.532216109702617,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.215266990585222,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.12726314162154,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0913353966418751,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.480619955315749,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 1.53846153846154,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 5.86529706898154,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.00482712351127131,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.324454658120818,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 7.79978332803809,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.633516171670183,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.04718125131449,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_27 = XGBTree(tree_dict_0_27, learning_rate=0.05)
        trees_output_0.append(tree_0_27)
        totalTrees += 1
        tree_dict_0_28 = {
            'feature': 0,
            'threshold': 4.6044271190637,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 0.131513193480171,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.486325670509,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.6060764852775,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.282741898055688,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.29855399024516,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0740340292437502,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.280076155547154,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': -1.6391198945976,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.06181041939996,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.00943845921811451,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.364446651680256,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.86021616487757,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.496924375064617,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.941423381327085,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_28 = XGBTree(tree_dict_0_28, learning_rate=0.05)
        trees_output_0.append(tree_0_28)
        totalTrees += 1
        tree_dict_0_29 = {
            'feature': 0,
            'threshold': 4.94530777859353,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 4.9263925116719,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -2.62124478437458,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.477288025884058,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.137199666516534,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.57850116873668,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.158323113963611,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.543641618867123,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 5.33674070250706,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.92520465322592,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.250292162692789,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.617146066412599,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.26195987784744,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.750094727527811,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.30829865248167,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_29 = XGBTree(tree_dict_0_29, learning_rate=0.05)
        trees_output_0.append(tree_0_29)
        totalTrees += 1
        tree_dict_0_30 = {
            'feature': 0,
            'threshold': 1.801019341746,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 5.49461885049399,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.70811442417713,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.502482290965647,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.21017868621505,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -4.02664443764407,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0949235870892413,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.416566939448285,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': -0.387941920460508,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.97060443054474,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0391255902556289,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.359870517365463,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 6.91974423682692,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.405809294277565,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.756878158532903,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_30 = XGBTree(tree_dict_0_30, learning_rate=0.05)
        trees_output_0.append(tree_0_30)
        totalTrees += 1
        tree_dict_0_31 = {
            'feature': 0,
            'threshold': 1.79564363778407,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 1.04266689115854,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -5.10466118805924,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.566648662205105,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.265697846901334,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.57850116873668,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.000871529338696291,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.447358453556428,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 0.265424442233591,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.92520465322592,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0151076687563956,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.466667175065511,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.67008433144795,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.505288344272448,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.01339333121232,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_31 = XGBTree(tree_dict_0_31, learning_rate=0.05)
        trees_output_0.append(tree_0_31)
        totalTrees += 1
        tree_dict_0_32 = {
            'feature': 0,
            'threshold': 1.801019341746,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 0.411023034750914,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -5.84118836444738,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.56878342123681,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.31168672387979,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.10136925168803,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0694719085806505,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.315129402091301,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': -0.497656757952464,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -6.09767371412576,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0750660747549249,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.169305971030268,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.67008433144795,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.449923737084236,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.958133001174292,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_32 = XGBTree(tree_dict_0_32, learning_rate=0.05)
        trees_output_0.append(tree_0_32)
        totalTrees += 1
        tree_dict_0_33 = {
            'feature': 0,
            'threshold': 1.89149307770406,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 1.69445304213035,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -4.19835016300397,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.505144855338836,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.265709454710941,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.87504389410531,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0737495625413443,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.208323481945311,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 2.36090538367564,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.87016756955598,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0309051009079092,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.421793308924574,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.3241438686729,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.472505708336701,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.871419255682268,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_33 = XGBTree(tree_dict_0_33, learning_rate=0.05)
        trees_output_0.append(tree_0_33)
        totalTrees += 1
        tree_dict_0_34 = {
            'feature': 2,
            'threshold': 1.44931643566741,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -0.89098170965654,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -2.79272535971964,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.462262274956725,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.247417817549396,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 6.26488321814601,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0928170088295286,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.201805015659698,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 1.27497058756579,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 9.4391127946292,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0212248580845382,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.494428441323645,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.17015599375733,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.347875876321475,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.68790229036333,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_34 = XGBTree(tree_dict_0_34, learning_rate=0.05)
        trees_output_0.append(tree_0_34)
        totalTrees += 1
        tree_dict_0_35 = {
            'feature': 0,
            'threshold': 4.9337895619008,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 5.3979635553436,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -2.62124478437458,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.369117374834508,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.118911596702977,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -5.57070309769041,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0040389933679225,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.356766855434315,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 0.481299712964097,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.92520465322592,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0694494212761289,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.428754123069618,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.06202936635392,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.490368309777368,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.808682115501637,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_35 = XGBTree(tree_dict_0_35, learning_rate=0.05)
        trees_output_0.append(tree_0_35)
        totalTrees += 1
        tree_dict_0_36 = {
            'feature': 0,
            'threshold': 4.94530777859353,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 1.69445304213035,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -6.59263763847078,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.501924941497125,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.202476636245304,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.43068692626766,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0317628953759831,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.396027966807416,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.04020048222694,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -5.41822391299769,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0210999560112182,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.323187794088563,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.3241438686729,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.644676777042679,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.867065148407786,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_36 = XGBTree(tree_dict_0_36, learning_rate=0.05)
        trees_output_0.append(tree_0_36)
        totalTrees += 1
        tree_dict_0_37 = {
            'feature': 1,
            'threshold': 4.86209326940933,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 0.123091978484241,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.4320445153754,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.418283869966713,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.193583124690327,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 7.71025827803923,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0301801698009559,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.290891324430158,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 6.60225169614777,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': 9.23059735076212,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.140753946485339,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.410576311308387,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.63988932894891,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.616779144824078,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.03030540061848,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_37 = XGBTree(tree_dict_0_37, learning_rate=0.05)
        trees_output_0.append(tree_0_37)
        totalTrees += 1
        tree_dict_0_38 = {
            'feature': 0,
            'threshold': 1.801019341746,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 0.524904220109017,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -1.89334092313824,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.405229954482396,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.197075415615938,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 6.97979313498987,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0773340834477287,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.256880367521048,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 0.560538910639513,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 5.20439763168453,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.136815181404731,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.17957530653544,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.62952009020417,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.318837098461855,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.605513863837376,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_38 = XGBTree(tree_dict_0_38, learning_rate=0.05)
        trees_output_0.append(tree_0_38)
        totalTrees += 1
        tree_dict_0_39 = {
            'feature': 2,
            'threshold': -1.31713302705884,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 5.36170212846611,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -5.37335270054253,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.448099128261813,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.228162823094196,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': -8.48046034684537,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.244304557327355,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.123873626643624,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 3.73169134291799,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 6.97979313498987,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0655578714295052,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.288167099614258,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 8.31148560254624,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.343218258240333,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.660497617200924,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_39 = XGBTree(tree_dict_0_39, learning_rate=0.05)
        trees_output_0.append(tree_0_39)
        totalTrees += 1
        tree_dict_0_40 = {
            'feature': 0,
            'threshold': 1.801019341746,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 0.131513193480171,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -8.32147577656216,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.473452995388889,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.262665239608589,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -4.74202082184761,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.131696304242922,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.10606988049095,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.39514444340635,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.82857903912044,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0534856280527676,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.380654284749035,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.91882911190069,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.508512354694974,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.925927601898673,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_40 = XGBTree(tree_dict_0_40, learning_rate=0.05)
        trees_output_0.append(tree_0_40)
        totalTrees += 1
        tree_dict_0_41 = {
            'feature': 0,
            'threshold': 5.8153668949836,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': 0.202331176159658,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -3.38434137189319,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.399679480685383,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.130647491765269,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -1.22262326721378,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0522289291712424,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.220163116016036,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': 6.15384615384615,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.75681789355486,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.105982432497409,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.370611029760075,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 9.38815654318191,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.613576279092024,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 1.14816546428282,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_41 = XGBTree(tree_dict_0_41, learning_rate=0.05)
        trees_output_0.append(tree_0_41)
        totalTrees += 1
        tree_dict_0_42 = {
            'feature': 0,
            'threshold': -1.26174800669597,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -0.49409123614803,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -4.77501853772389,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.423180285233972,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.276955677606811,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.79862945348063,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0639369903668826,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.220340573246226,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': -2.76642484140978,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 8.72596920546806,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.139160696658152,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.307957318657453,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.63988932894891,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.227759388009594,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.655957764936437,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_42 = XGBTree(tree_dict_0_42, learning_rate=0.05)
        trees_output_0.append(tree_0_42)
        totalTrees += 1
        tree_dict_0_43 = {
            'feature': 2,
            'threshold': 6.2316215676096,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 6.16170157255703,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': -4.84492640320587,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.345282607607962,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0828511082466805,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': -8.48046034684537,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.190130322377458,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.22181924820614,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 5.94637746880675,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 8.51079512416107,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.141504622213496,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.36795654640557,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.88776633023963,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.560293523244869,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.901360550535099,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_43 = XGBTree(tree_dict_0_43, learning_rate=0.05)
        trees_output_0.append(tree_0_43)
        totalTrees += 1
        tree_dict_0_44 = {
            'feature': 0,
            'threshold': 4.9337895619008,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -3.77997874278582,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -5.91733549635263,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.459639830131698,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.233804084017494,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -3.85972891925564,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.148293914814353,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0878735475831404,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 8.32603936171002,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -2.46611913837642,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0938504913026304,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.330003117634061,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.86067445535465,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.509816547605474,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.880035457964223,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_44 = XGBTree(tree_dict_0_44, learning_rate=0.05)
        trees_output_0.append(tree_0_44)
        totalTrees += 1
        tree_dict_0_45 = {
            'feature': 0,
            'threshold': 0.729174752550255,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 5.65665790661026,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -6.06519231241189,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.293849773804254,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.127124502351602,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -5.43094875872449,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0220967448171674,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.224557130783168,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.49644342286868,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.58588061958475,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.00435861671661156,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.30489220230337,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 8.63988932894891,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.353871269588872,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.662509346333667,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_45 = XGBTree(tree_dict_0_45, learning_rate=0.05)
        trees_output_0.append(tree_0_45)
        totalTrees += 1
        tree_dict_0_46 = {
            'feature': 2,
            'threshold': 2.11202950565272,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 6.16170157255703,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -6.59617416801955,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.364320844490393,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.162431158819214,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': -8.48046034684537,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.117051855646777,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.196699652559227,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': 6.68073806681258,
                'value': 0,
                'left': {
                    'feature': 2,
                    'threshold': 7.75086299456047,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0223003156343655,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.269191460790216,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 2,
                    'threshold': 9.88776633023963,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.448898580992997,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.788580435325547,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_46 = XGBTree(tree_dict_0_46, learning_rate=0.05)
        trees_output_0.append(tree_0_46)
        totalTrees += 1
        tree_dict_0_47 = {
            'feature': 0,
            'threshold': 5.03447152248391,
            'value': 0,
            'left': {
                'feature': 2,
                'threshold': -1.77798943833406,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -2.12212726515036,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.344381880782508,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.156900365064468,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -5.92005395431685,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.156767826637325,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0739394174584484,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 2,
                'threshold': -0.571174112567409,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 9.13068207126401,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0317385415117474,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.321674458800536,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 9.59027986095136,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.345624350501479,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.640689041715232,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_47 = XGBTree(tree_dict_0_47, learning_rate=0.05)
        trees_output_0.append(tree_0_47)
        totalTrees += 1
        tree_dict_0_48 = {
            'feature': 0,
            'threshold': 0.887491598331754,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': 4.5709774113479,
                'value': 0,
                'left': {
                    'feature': 1,
                    'threshold': -4.11340736309347,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.315606588228109,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.131520161573943,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': -8.27948530857058,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.135478782901021,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.13490970410519,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': -1.83906082192424,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 7.06181041939996,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.0734906996285076,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.159172927260841,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.3241438686729,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.206113326037222,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.488329575038782,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_48 = XGBTree(tree_dict_0_48, learning_rate=0.05)
        trees_output_0.append(tree_0_48)
        totalTrees += 1
        tree_dict_0_49 = {
            'feature': 0,
            'threshold': -2.77371955251413,
            'value': 0,
            'left': {
                'feature': 1,
                'threshold': -2.01869466324691,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': -8.43289409906565,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.414218502247879,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.265252208637172,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 1,
                    'threshold': 9.07257688203802,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.109632445295918,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.188045323731058,
                        'left': None,
                        'right': None
                    }
                }
            },
            'right': {
                'feature': 1,
                'threshold': -5.06203625016888,
                'value': 0,
                'left': {
                    'feature': 0,
                    'threshold': 6.91974423682692,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': -0.193898098376638,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.0795210985833235,
                        'left': None,
                        'right': None
                    }
                },
                'right': {
                    'feature': 0,
                    'threshold': 7.8831469061984,
                    'value': 0,
                    'left': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.111480581023192,
                        'left': None,
                        'right': None
                    },
                    'right': {
                        'feature': -1,
                        'threshold': 0,
                        'value': 0.406548662556838,
                        'left': None,
                        'right': None
                    }
                }
            }
        }
        tree_0_49 = XGBTree(tree_dict_0_49, learning_rate=0.05)
        trees_output_0.append(tree_0_49)
        totalTrees += 1
        self.trees.append(trees_output_0)
        
        print(f'XGBoost Model initialized with {self.output_count} outputs')
        print(f'Total trees: {totalTrees}')
        
    def predict(self, inputs):
        inputs = np.array(inputs, dtype=float)
        
        if inputs.shape != (self.input_dim,):
            raise ValueError(f'Input must be a vector of size {self.input_dim}. Got shape {inputs.shape}')
        
        outputs = []
        
        for o in range(self.output_count):
            prediction = self.biases[o]
            
            if o < len(self.trees):
                for tree in self.trees[o]:
                    prediction += tree.predict(inputs)
            
            outputs.append(float(prediction))
        
        return outputs

    def predict_batch(self, X):
        X = np.array(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f'Input must be a matrix with {self.input_dim} columns')
        
        predictions = np.zeros((X.shape[0], self.output_count))
        
        for i in range(X.shape[0]):
            predictions[i] = self.predict(X[i])
        
        return predictions

    def get_model_info(self):
        info = {
            'input_dim': self.input_dim,
            'output_count': self.output_count,
            'total_trees': sum(len(t) for t in self.trees),
            'biases': self.biases,
            'trees_per_output': [len(t) for t in self.trees]
        }
        return info

# Example usage:
if __name__ == '__main__':
    model = XGBoostModel()
    test_input = [1.0, 1.0, 1.0]
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
        
        
     
    
    
    
    
