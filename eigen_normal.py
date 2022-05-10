#-*- coding: utf-8 -*-
# Time-stamp: <2018-01-03 14:51:48 shunta>
"Output eigenvalue"
import hamiltonian
import numpy as np
import itertools
import sys
import os
from multiprocessing import Pool
from time import time

DIM = 8 # Hamiltonianの次元
nMax = 20

def output_fermisurface(n, t, alpha, mu):
    '''
    Output eigenvalues around K point
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    nx = int(n / (nMax ** 2))
    ny = int(n / nMax) % nMax
    nz = n % nMax
    k = np.array([2 * nx / nMax + 3.3, 2 * ny / nMax - 1, 2 * nz / nMax - 1])

    # Hamiltonianを定義・対角化
    hamil = hamiltonian.Hamiltonian()
    hamil.set_hamiltonian_normal(k, t, alpha, mu)
    eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian_normal())
    eigen_min = np.min(np.absolute(eigens.real))

    # 波数・エネルギー固有値を出力する
    if (eigen_min < 0.01):
        return k[0], k[1], k[2], eigen_min

def wrapper_output_fermisurface(args):
    return output_fermisurface(*args)

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

    # データを出力するためのフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")

    # 並列化の準備
    args = [[n, t, alpha, mu] for n in range(0, nMax ** 3)]
    p = Pool(processes=2)

    # データ出力
    start = time()
    results = p.map(wrapper_output_fermisurface, args)
    results = np.array([x for x in results if x is not None])

    np.savetxt("data/" + filename + ".d", results, delimiter="  ", fmt=["%.8f", "%.8f", "%.8f", "%.8f"])

    end = time()
    print("total time:", end - start)

except IndexError as IE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu"')

except ValueError as VE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu"')
