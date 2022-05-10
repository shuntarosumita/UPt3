#-*- coding: utf-8 -*-
# Time-stamp: <2017-05-28 17:49:04 shunta>
"Output eigenvalue in kx-ky plane"
import hamiltonian
import numpy as np
import itertools
import sys
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time

DIM = 8 # Hamiltonianの次元

def eigen_kxky(nx, t, alpha, mu, order, eta, delta, kz):
    '''
    Output eigenvalues on kz-kx plane
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    k = np.array([0, 0, kz])
    nMax = 200
    hamil = hamiltonian.Hamiltonian()

    k[0] = 2 * nx / nMax + 3.3

    result = np.empty((0, 4))
    for ny in range(0, nMax):
        k[1] = 2 * ny / nMax - 1

        # Hamiltonianを定義・対角化
        hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
        eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
        eigens = np.sort(eigens.real)

        result = np.append(result, np.array( [[k[0], k[1], k[2], eigens[int(DIM / 2)]]] ), axis=0)

    # 波数・エネルギー固有値を出力する
    return result

def wrapper_eigen_kxky(args):
    return eigen_kxky(*args)

# Main
try:
    filename = sys.argv[1]
    parameters = filename.split("_")

    # パラメータの設定
    if parameters[1] == "G-FS":
        t = np.array([1, 4, 1])                 # G-FS hopping integral
    elif parameters[1] == "A-FS":
        t = np.array([1, -4, 1])                # A-FS hopping integral
    elif parameters[1] == "K-FS":
        t = np.array([1, -1, 0.4])              # K-FS hopping integral
    else:
        raise ValueError

    alpha = np.array([float(parameters[2]), 0]) # antisymmetric spin-orbit coupling
    mu = float(parameters[3])                   # chemical potential
    order = float(parameters[4])                # order parameter
    eta = float(parameters[5])                  # ratio between Gamma[0] and Gamma[1]
    delta = float(parameters[6])                # intra p-wave order parameter
    kz = float(parameters[7])                   # wavevector kz

    # 並列化の準備
    args = [[i, t, alpha, mu, order, eta, delta, kz] for i in range(0, 200)]
    p = Pool(processes=2)

    # データ出力
    start = time()
    results = p.map(wrapper_eigen_kxky, args)
    end = time()
    print("total time:", end - start)

    results = np.array(results).T
    plt.contourf(results[0], results[1], results[3] / order, levels=np.linspace(0, 0.4, 500))
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.colorbar()
    plt.show()

    sys.exit()

except IndexError as IE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta_kz"')

except ValueError as VE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta_kz"')
