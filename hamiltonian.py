#-*- coding: utf-8 -*-
# Time-stamp: <2019-04-02 12:38:06 shunta>
"Hamiltonian of UPt3"
import math
import numpy as np

DIM = 8 # Hamiltonianの次元

# 計算に必要な行列・ベクトル
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, - 1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, - 1]])
vec_e = np.array([[1, 0], [- 0.5, 0.5 * math.sqrt(3)], [- 0.5, - 0.5 * math.sqrt(3)]]).T
vec_r = np.array([[0.5, 1 / (2 * math.sqrt(3))], [- 0.5, 1 / (2 * math.sqrt(3))], [0, - 1 / math.sqrt(3)]]).T

class Hamiltonian():
    "Hamiltonian class"
    def __init__(self, k=np.zeros(3), t=np.zeros(3), alpha=np.zeros(2), mu=0, order=0, eta=0, delta=0):
        self.set_hamiltonian(k, t, alpha, mu, order, eta, delta)
        self.set_hamiltonian_normal(k, t, alpha, mu)
    def __str__(self):
        return str(self.__hamil)
    def __repr__(self):
        return "Hamiltonian(" + str(self.__hamil) + ",\n" + str(self.__hamil_normal) + ")"

    def get_hamiltonian(self):
        "Getter of __hamil"
        return self.__hamil
    def set_hamiltonian(self, k, t, alpha, mu, order, eta, delta):
        "Setter of __hamil"
        offdiagonal = self.order_part(k, order, eta, delta)
        self.__hamil = np.r_[np.c_[self.normal_part(k, t, alpha, mu), offdiagonal], \
                             np.c_[np.conj(offdiagonal.T), - (self.normal_part(-k, t, alpha, mu)).T]]

    # 属性 __hamil にゲッター・セッターを設定
    hamil = property(get_hamiltonian, set_hamiltonian)

    def get_hamiltonian_normal(self):
        "Getter of __hamil_normal"
        return self.__hamil_normal
    def set_hamiltonian_normal(self, k, t, alpha, mu):
        "Setter of __hamil_normal"
        self.__hamil_normal = self.normal_part(k, t, alpha, mu)

    # 属性 __hamil_normal にゲッター・セッターを設定
    hamil_normal = property(get_hamiltonian_normal, set_hamiltonian_normal)

    def intra(self, k, t):
        "Intrasublattice hopping"
        k_para = np.delete(k, 2)
        return (2 * t[0] * np.sum(np.cos(np.dot(k_para, vec_e)))) + (2 * t[1] * math.cos(k[2]))

    def inter(self, k, t):
        "Intersublattice hopping"
        k_para = np.delete(k, 2)
        return 2 * t[2] * math.cos(0.5 * k[2]) * np.sum(np.exp(1j * np.dot(k_para, vec_r)))

    def g1(self, k):
        "g-vector 1"
        k_para = np.delete(k, 2)
        return np.array([0, 0, np.sum(np.sin(np.dot(k_para, vec_e)))])

    def g2(self, k):
        "g-vector 2"
        k_para = np.delete(k, 2)
        calc = np.dot(k_para, vec_e)
        return np.array([np.sum(k_para[0] * math.sin(k[2]) * np.cos(calc)), \
                         np.sum(k_para[1] * math.sin(k[2]) * np.cos(calc)), \
                         np.sum(- math.cos(k[2]) * np.sin(calc))])

    def normal_part(self, k, t, alpha, mu):
        "Normal part Hamiltonian"
        Pauli = np.array([sigma_x, sigma_y, sigma_z]).reshape(3, 4)
        mat = np.kron((self.intra(k, t) - mu) * np.eye(2), np.eye(2)) \
              + np.kron(self.inter(k, t).real * np.eye(2), sigma_x) \
              - np.kron(self.inter(k, t).imag * np.eye(2), sigma_y) \
              + np.kron(np.sum((alpha[0] * self.g1(k) + alpha[1] * self.g2(k)).reshape(3, 1) \
                               * Pauli, axis=0).reshape(2, 2), sigma_z)

        # unitary変換をして周期性を回復
        U = np.kron(np.eye(2), np.diag( [1, np.exp(1j * np.dot(k, np.array([0, - 1 / math.sqrt(3), 0.5])))] ))
        mat = np.dot(np.dot(U, mat), np.conj(U.T))

        return mat

    def order_intra_p(self, k):
        "Intrasublattice p-wave order parameter"
        k_para = np.delete(k, 2)
        # p_x, p_y
        p = np.sum(vec_e * np.sin(np.dot(k_para, vec_e)), axis=1)

        return np.array([np.kron(- p[0] * sigma_z - 1j * p[1] * np.eye(2), np.eye(2)), \
                         np.kron(- p[1] * sigma_z + 1j * p[0] * np.eye(2), np.eye(2))])

    def order_inter_p(self, k):
        "Intersublattice p-wave order parameter"
        k_para = np.delete(k, 2)
        # p_x, p_y
        p = np.sum(vec_r * np.exp(1j * np.dot(k_para, vec_r)), axis=1) \
            * (- 1j) * math.sqrt(3) * math.cos(0.5 * k[2])

        return np.array([np.kron(- p[0].real * sigma_z - 1j * p[1].real * np.eye(2), sigma_x) \
                         + np.kron(p[0].imag * sigma_z + 1j * p[1].imag * np.eye(2), sigma_y), \
                         np.kron(- p[1].real * sigma_z + 1j * p[0].real * np.eye(2), sigma_x) \
                         + np.kron(p[1].imag * sigma_z - 1j * p[0].imag * np.eye(2), sigma_y)])

    def order_inter_d_f(self, k):
        "Intersublattice d- and f-wave order parameter"
        k_para = np.delete(k, 2)
        calc = np.sum(vec_r * np.exp(1j * np.dot(k_para, vec_r)), axis=1) \
               * (- math.sqrt(3)) * math.sin(0.5 * k[2])
        d = calc.imag # d_{xz}, d_{yz}
        f = calc.real # f_{xyz}, f_{(x^2 - y^2)z}

        return np.array([np.kron(f[1] * sigma_x, sigma_x) - np.kron(d[1] * sigma_x, sigma_y), \
                         np.kron(f[0] * sigma_x, sigma_x) - np.kron(d[0] * sigma_x, sigma_y)])

    def order_part(self, k, order, eta, delta):
        "Order parameter part Hamiltonian"
        Gamma = delta * self.order_intra_p(k) + 0.2 * self.order_inter_p(k) + self.order_inter_d_f(k)
        mat = (Gamma[0] + Gamma[1] * 1j * eta) * order / math.sqrt(1 + eta ** 2)

        # unitary変換をして周期性を回復
        U = np.kron(np.eye(2), np.diag( [1, np.exp(1j * np.dot(k, np.array([0, - 1 / math.sqrt(3), 0.5])))] ))
        mat = np.dot(np.dot(U, mat), np.conj(U.T))

        return mat
