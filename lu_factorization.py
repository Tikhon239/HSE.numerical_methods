import numpy as np
from scipy.linalg import lu
#лекция 2, слайд 13
#работает в предположении того, что L[i, i] != 0
def TeaShyLU(A):
    L = np.zeros(A.shape)
    U = np.eye(A.shape[0]) #используем знаниние о том, что U[i, i] = 1
    for i in range(A.shape[0]):
        for k in range(i+1):
            L[i, k] = A[i, k] - L[i, :k].dot(U[:k, k]) #k <= i
        for k in range(i+1, A.shape[0]):
            U[i, k] = (A[i, k] - L[i, :i].dot(U[:i, k]))/L[i, i] #k > i
    return  L, U

A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
p, l, u = lu(A)
#p - матрица перестановок, тоже самое как в гауссе, когда меняем строки, чтобы на позии a[i,i] стоял макисмальный элемент в столбце a[:, i]
A = np.linalg.inv(p).dot(A) 
p, l, u = lu(A)
print(np.linalg.norm(l.dot(u) - A, ord = np.inf))
L, U = TeaShyLU(A)
print(np.linalg.norm(L.dot(U) - A, ord = np.inf))