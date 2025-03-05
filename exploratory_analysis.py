import time
import numpy as np
from triangle import TrilocalModel
from triangle_inequalities import valid_distribution

c_alpha, c_beta, c_gamma = 6, 6, 6

N = 150
N_trials = 200
delta3 = 2/N
delta2 = 4/3/N

e3 = np.arange(-1, 1+delta3, delta3)
e2 = np.arange(-1/3, 1+delta2, delta2)

E3, E2 = np.meshgrid(e3, e2, indexing='ij')
E1 = np.empty_like(E3)

model = TrilocalModel.uniform(c_alpha, c_beta, c_gamma, 2, 2, 2)
dof = model.degrees_of_freedom()
models = np.zeros(shape=E3.shape+(dof, ))
error = np.inf * np.ones(E3.shape)

a, b, c = np.meshgrid((1,-1), (1,-1), (1,-1), indexing='ij')

for i, e3_ in enumerate(e3):
    for j, e2_ in enumerate(e2):
        # The equation that characterizes the plane to be explored
        e1_ = 1 + e3_ - e2_
        E1[i, j] = e1_
        if not valid_distribution(e1_, e2_, e3_):
            continue
        print(f'{e3_} {e2_}')
        p = 1/8*(1 + e1_*(a+b+c) + e2_*(a*b+a*c+b*c) + e3_*a*b*c)
        model = model.optimize(p, number_of_trials=N_trials)
        error[i,j] = np.sqrt(model.cost(p) / 8)
        models[i,j] = model.optimizer_representation()

filename = (f'Exploratory_analysis_{c_alpha}{c_beta}{c_gamma}_{N_trials}_trials_'
            + time.strftime("%Y%m%d-%H%M%S") + '.npz')
np.savez(filename, e3=e3, e2=e2, E3=E3, E2=E2, E1=E1, models=models, error=error)
