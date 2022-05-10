#-*- coding: utf-8 -*-
# Time-stamp: <2017-05-19 17:43:12 shunta>
"Calculation of superconducting dispersion in UPt3"
import output
import sys
import numpy as np
import os
from multiprocessing import Pool
from time import time

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

    # 並列化の準備
    args = [[i, t, alpha, mu, order, eta, delta] for i in range(0, 4000)]
    p = Pool(processes=2)

    ##### テスト用 #####
    if parameters[0] == "test":
        results = p.map(output.wrapper_test, args)
        results = np.array(results).T

        import matplotlib.pyplot as plt
        for i in range(1, results.shape[0]):
            plt.plot(results[0], results[i] / order)
        plt.ylim(-0.2, 0.2)
        plt.show()
        sys.exit()
    ##### テスト用 #####

    # データを出力するためのフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")
    fout = open("data/" + filename + ".d", "wt")

    start = time()

    # 出力内容を指定してデータ出力
    if parameters[0] == "eigenKH":
        results = p.map(output.wrapper_eigen_KH, args)
        for result in results:
            for out in result:
                print(out, file=fout, end="  ")
            print("", file=fout)

    elif parameters[0] == "gapkz":
        results = p.map(output.wrapper_gap_kz, args)
        for result in results:
            print(result[0], result[1], file=fout, sep="  ")

    elif parameters[0] == "Chern":
        results = p.map(output.wrapper_Chern_num, args)
        for result in results:
            if result[1] == result[1]: # Nan check
                print(result[0], result[1], result[2], file=fout, sep="  ")

    else:
        raise ValueError

    end = time()
    print("total time:", end - start)

    fout.close()

except IndexError as IE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta"')

except ValueError as VE:
    print('Usage: "python', sys.argv[0], 'Calctype_FS_alpha_mu_order_eta_delta"')
