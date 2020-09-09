import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""

    def __init__(self, x, n_components=None):
        self.x = x
        self.dimension = x.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)                           #矩阵转置
        x_cov = np.cov(x_T)                                  #协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""

        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)), b))
        c = np.real(c)
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort_values(by=0, ascending=False)
        return c_df_sort

    def explained_varience_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]

    def paint_varience_(self):
        explained_variance_ = self.explained_varience_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        varience = self.explained_varience_()

        if self.n_components:  # 指定降维维度
            p = np.transpose(c_df_sort.values[0:self.n_components, 1:])
            y = np.dot(self.x, p)
            return y, p

        varience_sum = sum(varience)
        varience_radio = varience / varience_sum

        varience_contribution = 0
        for R in range(self.dimension):
            varience_contribution += varience_radio[R]
            if varience_contribution >= 0.99:
                break

        p = np.transpose(c_df_sort.values[0:R + 1, 1:])  # 取前R个特征向量
        y = np.dot(self.x, p)
        return y, p
