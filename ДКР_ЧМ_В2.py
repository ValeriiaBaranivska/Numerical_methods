import numpy as np
from tabulate import tabulate
import csv
import pandas as pd
"""
ДКР. Чисельні методи
Варіант №2
Баранівська Валерія  
студентка групи КМ-23
"""

def f(x):
    return np.sin(x) - (x - 1) ** 2

def df(x):
    return np.cos(x) - 2 * (x - 1)

# ------------------  метод хорд  ------------------
def secant_method_detailed_with_steps(x0, x1, tol=1e-4, max_iter=100):
    results = []
    prev_xk = x1  # Для обчислення різниці з попереднім значенням x_k
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < tol:
            break
        xk = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        abs_diff = abs(xk - prev_xk)  # Різниця між x_k і попереднім значенням x_(k-1)
        prev_xk = xk
        results.append([i, x0, f_x0, x1, f_x1, xk, f(xk), abs_diff < tol])
        if abs_diff < tol:
            break
        x0, x1 = x1, xk
    return results

# ------------------  метод дотичних  ------------------
def newton_method_detailed_with_steps(x0, tol=1e-4, max_iter=100):
    results = []
    for i in range(max_iter):
        f_x0 = f(x0)
        df_x0 = df(x0)
        if abs(df_x0) < tol:
            print("Похідна близька до нуля. Метод не працює.")
            break
        x1 = x0 - f_x0 / df_x0
        abs_diff = abs(x1 - x0)
        results.append([x0, f_x0, df_x0, x1, abs_diff < tol])
        if abs_diff < tol:
            break
        x0 = x1
    return results

# ------------------  метод дихотомії  ------------------
def bisection_method_detailed_with_steps(a, b, tol=1e-4, max_iter=100):
    results = []
    if f(a) * f(b) > 0:
        return None  # Немає зміни знака на інтервалі
    for i in range(max_iter):
        x_star = (a + b) / 2
        abs_diff = abs(b - a)  # Обчислюємо різницю між b і a
        results.append([i, a, f(a), b, f(b), x_star, f(x_star), abs_diff < tol])  # Додаємо до таблиці
        if abs_diff < tol:  # Якщо різниця менша за допустиму похибку
            break
        if f(x_star) * f(a) < 0:
            b = x_star
        else:
            a = x_star
    return results

# ------------------  метод простих ітерацій ------------------
def simple_iteration_method_detailed_with_steps(x0, tol=1e-4, max_iter=100):
    def phi(x):
        return np.sin(x) + 1
    results = []
    for i in range(max_iter):
        x1 = phi(x0)
        abs_diff = abs(x1 - x0)  # Обчислюємо різницю між x_(k+1) та x_k
        results.append([i, x0, x1, abs_diff < tol])  # Додаємо до таблиці
        if abs_diff < tol:  # Якщо різниця менша за допустиму похибку
            break
        x0 = x1
    return results

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
    #виведення округлених результатів
    print("Розв'язок системи:", np.round(x,4))
    print("Прогоночні коефіцієнти Альфа:", np.round(alpha_i,4))
    print("Прогоночні коефіцієнти Гамма:", np.round(gamma_i,4))

def _main_():
    # ------------------  завдання 1  ------------------
    # обчислення кожним методом у заданих умовах
    secant_steps_detailed = secant_method_detailed_with_steps(0, 2)
    newton_steps_detailed = newton_method_detailed_with_steps(0)
    newton_steps_detailed_2 = newton_method_detailed_with_steps(2)
    bisection_steps_detailed = bisection_method_detailed_with_steps(0, 1)
    simple_iteration_steps_detailed = simple_iteration_method_detailed_with_steps(0)
    with open("secant_method_results.csv", "w", newline='') as file:
        writer = csv.writer(file)
        # Запис заголовків
        writer.writerow(["i", "x0", "f(x0)", "x1", "f(x1)", "x_k", "f(x_k)", "|x_k - x_(k-1)| < tol"])
        # Запис даних
        writer.writerows(secant_steps_detailed)

    with open("newt.csv", "w", newline='') as file:
        writer = csv.writer(file)
        # Запис заголовків
        writer.writerow(["x_(k-1)", "f(x_(k-1))", "f'(x_(k-1))", "x_k", "|x_k - x_(k-1)| < e"])
        # Запис даних
        writer.writerows(newton_steps_detailed)

    with open("bisc.csv", "w", newline='') as file:
        writer = csv.writer(file)
        # Запис заголовків
        writer.writerow(["i","a", "f(a)", "b", "f(b)", "x* = (a+b)/2", "f(x*)", "|b - a| < e"])
        # Запис даних
        writer.writerows(bisection_steps_detailed)

    with open("simp.csv", "w", newline='') as file:
        writer = csv.writer(file)
        # Запис заголовків
        writer.writerow(["i","x_k", "x_(k+1) = phi(x_k)", "|x_(k+1) - x_k| < e"])
        # Запис даних
        writer.writerows(simple_iteration_steps_detailed)

    # Створимо і виведемо таблиці для кожного методу
    secant_table_detailed = tabulate(secant_steps_detailed,
                                     headers=["i","a", "f(a)", "b", "f(b)", "x_k", "f(x_k)", "|x_k - x_(k-1)| < e"],
                                     tablefmt="grid")
    newton_table_detailed = tabulate(newton_steps_detailed, headers=["x_(k-1)", "f(x_(k-1))", "f'(x_(k-1))", "x_k", "|x_k - x_(k-1)| < e"], tablefmt="grid")

    newton_table_detailed_2 = tabulate(newton_steps_detailed_2,
                                     headers=["x_(k-1)", "f(x_(k-1))", "f'(x_(k-1))", "x_k", "|x_k - x_(k-1)| < e"],
                                     tablefmt="grid")

    bisection_table_detailed = tabulate(bisection_steps_detailed, headers=["i","a", "f(a)", "b", "f(b)", "x* = (a+b)/2", "f(x*)", "|b - a| < e"], tablefmt="grid")

    simple_iteration_table_detailed = tabulate(simple_iteration_steps_detailed, headers=["i","x_k", "x_(k+1) = phi(x_k)", "|x_(k+1) - x_k| < e"], tablefmt="grid")

    print(tabulate([["  Метод хорд  "]], tablefmt="fancy_grid"))
    print(secant_table_detailed)

    print(tabulate([["  Метод  дотичних. Метод Ньютона  "]], tablefmt="fancy_grid"))
    print(newton_table_detailed)

    print(tabulate([["  Метод  дотичних. Метод Ньютона 22222 "]], tablefmt="fancy_grid"))
    print(newton_table_detailed_2)

    print(tabulate([["  Метод дихотомії. Поділ відрізку навпіл  "]], tablefmt="fancy_grid"))
    print(bisection_table_detailed)

    print(tabulate([["  Метод простих ітерацій   "]], tablefmt="fancy_grid"))
    print(simple_iteration_table_detailed)

    #умови до завлання 2,3,4
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
    L, U = LU_factorization(A) #обчилення матриць
    #виведення округлених результатів
    print("     Матриця L:")
    print(np.round(L,4))

    print("\n   Матриця U:")
    print(np.round(U,4))

    # Перевірка, що A = L * U
    A_reconstructed = np.dot(L, U)
    print("\n   Відновлена матриця A:")
    print(A_reconstructed)

    print("\n   Визначник матриці А = ", round(det(L),4))

    # ------------------ метод Гаусса  ------------------
    print(tabulate([["    Метод Гаусса    "]], tablefmt="fancy_grid"))
    # Обчислення оберненої матриці
    inv_a = inverse(A)
    print(" Обернена матриця:")
    print(np.round(inv_a,3))

    # Обчислення F_0 = E - A * D0
    F_0 = np.dot(np.subtract(np.eye(4), A), inv_a)

    print("\n Результат F_0:")
    print(np.round(F_0,3))

    # Обчислення D_1 = D_0 + D_0 * F_0
    D_1 = np.add(inv_a, np.dot(inv_a, F_0))
    print("\n Результат D_1:")
    print(np.round(D_1,3))

    # Норма
    norm_sub = np.dot(inv_a, F_0)
    norm = np.max(norm_sub)
    print("\n Результат нормування:")
    if abs(round(norm)) < 0.02:
        print(True)
    else:
        print(False)

    # ------------------ Метод прогонки  ------------------
    print(tabulate([["   Метод прогонки   "]], tablefmt="fancy_grid"))
    print(" Mатриця коефiцiєнтiв системи:\n", A_prog)
    method_progonki(A_prog, d_prog)


    # Завантажте CSV
    secant = pd.read_csv('secant_method_results.csv')
    bis = pd.read_csv('bisc.csv')
    newt = pd.read_csv('newt.csv')
    simp = pd.read_csv('simp.csv')

    # Експортуйте в LaTeX
    print(secant.to_latex(index=False))
    print(bis.to_latex(index=False))
    print(newt.to_latex(index=False))
    print(simp.to_latex(index=False))

_main_()
