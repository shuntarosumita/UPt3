#-*- coding: utf-8 -*-
# Time-stamp: <2018-01-03 11:45:50 shunta>
"Output eigenvalue"
import hamiltonian
import numpy as np
import itertools
import sys
import os
from multiprocessing import Pool
from time import time

DIM = 8 # Hamiltonianの次元
nMax = 200

def eigen_node(n, t, alpha, mu, order, eta, delta):
    '''
    Output eigenvalues around K point
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    nx = n % nMax
    ny = int(n / nMax) % nMax
    nz = int(n / (nMax ** 2))
    k = np.array([2 * nx / nMax + 3.3, 2 * ny / nMax - 1, 2 * nz / nMax - 1])

    # Hamiltonianを定義・対角化
    hamil = hamiltonian.Hamiltonian(k, t, alpha, mu, order, eta, delta)
    eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
    eigen_min = np.min(np.absolute(eigens.real))
    # eigens = np.sort(eigens.real)
    # eigen_min = eigens[int(DIM / 2)]

    # 波数・エネルギー固有値を出力する
    if (eigen_min < 0.005 * order):
        return n, k[0], k[1], k[2], eigen_min

def wrapper_eigen_node(args):
    return eigen_node(*args)

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

    # データを出力するためのフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")

    # 並列化の準備
    args = [[n, t, alpha, mu, order, eta, delta] for n in range(0, nMax ** 3)]
    p = Pool(processes=2)

    # データ出力
    start = time()
    results = p.map(wrapper_eigen_node, args)
    results = np.array([x for x in results if x is not None])#.reshape((-1, 5))

    np.savetxt("data/" + filename + ".d", results, delimiter="  ", fmt=["%.0f", "%.8f", "%.8f", "%.8f", "%.8f"])

    end = time()
    print("total time:", end - start)

except IndexError as IE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta"')

except ValueError as VE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta"')
