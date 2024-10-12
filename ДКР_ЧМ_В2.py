import numpy as np
from tabulate import tabulate

"""
ДКР. Чисельні методи
Варіант №2
Баранівська Валерія  
студентка групи КМ-23
"""

# ------------------  метод хорд  ------------------
def method_chord():
    print(1)


# ------------------  метод дотичних  ------------------
def method_dot():
    print(1 + 1)


# ------------------  метод дихотомії  ------------------
# ------------------  метод простих ітерацій ------------------


# ------------------  LU-факторизація  ------------------
def LU_factorization(A):
    n = len(A)
    U = np.eye(n)  # Створюємо нижньо-трикутну матрицю з одиницями на діагоналі
    L = np.zeros_like(A)  # Початково заповнюємо нулями верхньо-трикутну матрицю

    for i in range(n):
        # Заповнюємо нижню трикутну матрицю L
        for j in range(i, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(j)))
        # Заповнюємо верхню трикутну матрицю U
        for j in range(i + 1, n):
            U[i][j] = (1 / L[i][i]) * (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i)))
    return L, U

# Знаходження визначника detA=detL
def det(m):
    p = 1
    n = len(m)
    for i in range(n):
        for j in range(n):
            if (i == j):
                p *= m[i][j]
    return p

# ------------------ метод Гаусса  ------------------
def inverse(a):
    n = len(a)
    # Constructing the augmented matrix
    augmented_matrix = np.hstack((a, np.eye(n)))

    # Gaussian elimination to get the inverse
    for k in range(n):
        if abs(augmented_matrix[k][k]) < 1.0e-12:
            for i in range(k + 1, n):
                if abs(augmented_matrix[i][k]) > abs(augmented_matrix[k][k]):
                    augmented_matrix[[k, i]] = augmented_matrix[[i, k]]  # Swapping of rows
                    break
        pivot = augmented_matrix[k][k]
        if pivot == 0:
            print("This matrix is not invertible.")
            return None
        augmented_matrix[k] /= pivot
        for i in range(n):
            if i == k or augmented_matrix[i][k] == 0:
                continue
            factor = augmented_matrix[i][k]
            augmented_matrix[i] -= factor * augmented_matrix[k]

    # Extracting the inverse matrix from the augmented matrix
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix

# ------------------ Метод прогонки  ------------------
def method_progonki(A, d):
    n = len(A)
    x = np.zeros(n)
    # Ініціалізація списків для коофіцієнтів
    a = []
    b = []
    c = []

    # Заповнення масивів a, b, c
    for i in range(n):
        if i > 0:
            a.append(A[i][i - 1])  # Ліві коофіцієнти
        else:
            a.append(0)  # Перший лівій коофіцієнт завжди 0
        b.append(A[i][i])  # Головні коофіцієнти
        if i < n - 1:
            c.append(A[i][i + 1])  # Праві коофіцієнти
        else:
            c.append(0)  # Останній правий коофіцієнт завжди 0

    print("Ліві коофіцієнти (a):", a)
    print("Головні коофіцієнти (b):", b)
    print("Праві коофіцієнти (c):", c)
    flag = False
    # Перевірка умови для методу прогонки
    for i in range(1, n):
        if abs(b[i]) >= abs(a[i - 1]) + abs(c[i - 1]):
            flag = True
        else:
            return  # Вихід з функції, якщо умова не виконується

    # прямий хід
    alpha_i = np.zeros(n)
    gamma_i = np.zeros(n)

    # обчислення прогоночних коефіцієнтів
    alpha_i[0] = -c[0] / b[0]
    gamma_i[0] = d[0, 0] / b[0]
    for i in range(1, n):

        k = b[i] + a[i] * alpha_i[i - 1]
        if k == 0:
            print("Ділення на нуль!")
            return
        alpha_i[i] = -c[i] / k
        gamma_i[i] = (d[i, 0] - a[i] * gamma_i[i - 1]) / k
    # обернений хід
    x[-1] = gamma_i[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha_i[i] * x[i + 1] + gamma_i[i]

    print("Розв'язок системи:", x)
    print("Прогоночні коефіцієнти Альфа:", alpha_i)
    print("Прогоночні коефіцієнти Гамма:", gamma_i)

def _main_():
    A = np.array([[12.45, 3.93, 6.15, 2.85],
                  [5.88, 12.68, 11.67, 3.69],
                  [5.67, 10.83, 12.06, 3.42],
                  [3.33, 5.49, 2.52, 10.50]])

    A_prog = np.array([[13.7, -4.7, 0, 0],
                       [5.5, 15.5, -4.6, 0],
                       [0, 3.9, 12.5, -2.7],
                       [0, 0, 2.25, 10.2]])

    d_prog = np.array([[2.5],
                       [8.5],
                       [4.4],
                       [5.6]])

    # ------------------  LU-факторизація  ------------------
    print(tabulate([["  LU-факторизація. Метод Холецького   "]], tablefmt="fancy_grid"))
    L, U = LU_factorization(A)

    print("     Матриця L:")
    print(L)

    print("\n   Матриця U:")
    print(U)

    # Перевірка, що A = L * U
    A_reconstructed = np.dot(L, U)
    print("\n   Відновлена матриця A:")
    print(A_reconstructed)

    print("\n   Визначник матриці А = ", det(L))

    # ------------------ метод Гаусса  ------------------
    print(tabulate([["    Метод Гаусса    "]], tablefmt="fancy_grid"))
    # Обчислення оберненої матриці
    inv_a = inverse(A)
    print(" Обернена матриця:")
    print(inv_a)

    # Обчислення F_0 = E - A * D0
    F_0 = np.dot(np.subtract(np.eye(4), A), inv_a)

    print("\n Результат F_0:")
    print(F_0)

    # Обчислення D_1 = D_0 + D_0 * F_0
    D_1 = np.add(inv_a, np.dot(inv_a, F_0))
    print("\n Результат D_1:")
    print(D_1)

    # Норма
    norm = np.max(D_1)
    if norm < 0.02:
        print(True)
    else:
        print(False)

    # ------------------ Метод прогонки  ------------------
    print(tabulate([["   Метод прогонки   "]], tablefmt="fancy_grid"))
    print(" Mатриця коефiцiєнтiв системи:\n", A_prog)
    method_progonki(A_prog, d_prog)

_main_()