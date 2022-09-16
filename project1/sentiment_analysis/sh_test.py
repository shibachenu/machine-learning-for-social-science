from project1 import perceptron, perceptron_single_step_update
import numpy as np
feature_matrix = np.matrix([
    [-0.44600691, -0.48995308, 0.10672829, -0.23279704, 0.34922046, 0.49462688, 0.23413718, 0.24492144, 0.04536957,
     0.29240331],
    [-0.40341641, 0.18935338, 0.49288347, 0.12559316, -0.25548025, -0.21518377, -0.49443982, 0.42027353, 0.23472971,
     -0.44255133],
    [-0.12251824, -0.39090683, 0.4144166, 0.33532461, 0.40672453, 0.12083048, -0.47791662, -0.10031615, -0.4681782,
     -0.43967674],
    [0.30205014, 0.2009804, 0.24596844, -0.38371268, 0.30504946, 0.1322984, 0.06530228, -0.28564335, 0.40193607,
     0.10658088],
    [0.38805038, -0.35276849, -0.44475011, -0.34922998, 0.42465848, -0.21405048, -0.23617881, 0.48033352,
     0.25019747, 0.16710515]])

labels = np.array([-1, 1, 1, -1, -1])
T = 5

theta, theta0 = perceptron(feature_matrix, labels, T)