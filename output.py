#-*- coding: utf-8 -*-
# Time-stamp: <2018-01-17 19:08:00 shunta>
"Output functions"
import hamiltonian
import numpy as np
import os
import itertools

DIM = 8 # Hamiltonianの次元

def eigen_kzkx(t, alpha, mu, order, eta, delta, filename):
    '''
    Output eigenvalues on kz-kx plane
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
      filename: output filename
    '''
    k = np.zeros(3)
    nMax = 1600
    hamil = hamiltonian.Hamiltonian()

    # データを出力するためのフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")
    fout = open("data/" + filename + ".d", "wt")

    for nx in range(0, nMax + 1):
        for nz in range(0, nMax + 1):
            k[0] = 4 / 3 * np.pi * nx / nMax
            k[2] = np.pi * nz / nMax

            # Hamiltonianを定義・対角化
            hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
            eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
            eigens = np.sort(eigens.real)

            # 波数・エネルギー固有値を出力する
            print(k[0], k[1], k[2], file=fout, sep="  ", end="  ")
            print(eigens[int(DIM / 2)], eigens[int(DIM / 2) + 1], file=fout, sep="  ", end="\n")
        print("", file=fout, end="\n")

    fout.close()


def eigen_KH(nz, t, alpha, mu, order, eta, delta):
    '''
    Output eigenvalues along K-H line
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    k = np.array([4 / 3 * np.pi, 0, 0])
    nzMax = 4000
    hamil = hamiltonian.Hamiltonian()

    k[2] = 2 * np.pi * nz / nzMax - np.pi

    # Hamiltonianを定義・対角化
    hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
    eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
    eigens = np.sort(eigens.real)

    # 波数・エネルギー固有値を出力する
    result = np.array([k[2]])
    result = np.append(result, eigens)
    return result

def wrapper_eigen_KH(args):
    return eigen_KH(*args)


def gap_kz(nz, t, alpha, mu, order, eta, delta):
    '''
    Output minimum eigenvalue (gap) as a function of k_z
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    k = np.zeros(3)
    nzMax = 100
    nMax = 2000
    hamil = hamiltonian.Hamiltonian()

    k[2] = np.pi * nz / nzMax
    eigen_min = 100
    for nx, ny in itertools.product(range(0, nMax), repeat=2):
        k[0] = np.pi * (2 * nx / nMax - 1)
        k[1] = 2 * np.sqrt(3) / 3 * np.pi * (2 * ny / nMax - 1)

        # Hamiltonianを定義・対角化
        hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
        eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
        eigens = np.sort(eigens.real)

        # 正の最小エネルギー固有値を保存
        if(eigens[int(DIM / 2)] < eigen_min):
            eigen_min = eigens[int(DIM / 2)]

    # 波数・エネルギー固有値を出力する
    return np.array([k[2], eigen_min])

def wrapper_gap_kz(args):
    return gap_kz(*args)


def Chern_num(nz, t, alpha, mu, order, eta, delta):
    '''
    Output Chern number as a function of k_z
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    k = np.zeros(3)
    nMax = 400
    nzMax = 1000
    hamil = hamiltonian.Hamiltonian()

    # Chern数を kz の関数として出力
    k[2] = np.pi * nz / nzMax

    # 全ての(kx, ky)点の psi を計算
    psi = np.zeros((nMax, nMax, DIM, int(DIM / 2)), dtype=np.complex)
    for nx, ny in itertools.product(range(0, nMax), repeat=2):
        k[0] = np.pi * (2 * nx / nMax - 1)
        k[1] = 2 * np.sqrt(3) / 3 * np.pi * (2 * ny / nMax - 1)

        # Hamiltonianを定義・対角化
        hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
        eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())

        # 固有ベクトルを固有値の昇順に並べ替える
        index = np.argsort(eigens.real)
        # eigens = eigens[index]
        u_mat = (u_mat.T[index]).T

        # 負の固有値に対応する固有ベクトルのみを psi に代入
        psi[nx][ny] = np.hsplit(u_mat, 2)[0]

    # psi をもとにChern数 nu を計算
    nu = np.complex(0)
    Fmax = 0
    for nx, ny in itertools.product(range(0, nMax), repeat=2):
        # Link variable
        Ux_1 = np.linalg.det(np.dot(np.conj(psi[nx][ny].T), psi[(nx + 1) % nMax][ny]))
        Uy_1 = np.linalg.det(np.dot(np.conj(psi[(nx + 1) % nMax][ny].T), psi[(nx + 1) % nMax][(ny + 1) % nMax]))
        Ux_2 = np.linalg.det(np.dot(np.conj(psi[nx][(ny + 1) % nMax].T), psi[(nx + 1) % nMax][(ny + 1) % nMax]))
        Uy_2 = np.linalg.det(np.dot(np.conj(psi[nx][ny].T), psi[nx][(ny + 1) % nMax]))

        # 規格化
        Ux_1 /= np.linalg.norm(Ux_1)
        Uy_1 /= np.linalg.norm(Uy_1)
        Ux_2 /= np.linalg.norm(Ux_2)
        Uy_2 /= np.linalg.norm(Uy_2)

        # Field strength
        F = np.log(Ux_1 * Uy_1 * np.conj(Ux_2) * np.conj(Uy_2)) / (2j * np.pi)
        if np.abs(F.real) > Fmax:
            Fmax = np.abs(F.real)

        nu += F

    # 波数・Chern数を出力する
    return np.array([k[2], nu.real, Fmax])

def wrapper_Chern_num(args):
    return Chern_num(*args)


def test(n, t, alpha, mu, order, eta, delta):
    '''
    Output eigenvalues along K-H line
      t: hopping integral
      alpha: antisymmetric spin-orbit coupling
      mu: chemical potential
      order: order parameter
      eta: ratio between Gamma[0] and Gamma[1]
      delta: intra p-wave order parameter
    '''
    k = np.array([4 / 3 * np.pi, 0, 0.5715])
    nMax = 4000
    hamil = hamiltonian.Hamiltonian()

    # k[0] = 2 * n / nMax + 3.3
    k[1] = 2 * n / nMax - 1

    # Hamiltonianを定義・対角化
    hamil.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
    eigens, u_mat = np.linalg.eig(hamil.get_hamiltonian())
    eigens = np.sort(eigens.real)

    # 波数・エネルギー固有値を出力する
    # result = np.array([k[0]])
    result = np.array([k[1]])
    result = np.append(result, eigens)
    return result

def wrapper_test(args):
    return test(*args)
