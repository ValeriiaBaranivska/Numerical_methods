import numpy as np
from sympy import symbols, simplify, lambdify

x_values = np.array([1,2,3,4,5,6])
L_n = [1.206948961, 4.324077125, 6.578965206,8.354248899,9.831323978,11.1035869 ]

x_1 = 2.25
x_2 = 0.75
x = symbols('x')

lagrange_poly = 0
for i in range(len(x_values)):
    term = L_n[i]
    for j in range(len(x_values)):
        if i != j:
            term *= (x - x_values[j]) / (x_values[i] - x_values[j])
    lagrange_poly += term

simplified_lagrange_poly = simplify(lagrange_poly)
print("Поліном Лагранжа:")
print(simplified_lagrange_poly)

lagrange_func = lambdify(x, simplified_lagrange_poly, 'numpy')
lagrange_result_x_1 = lagrange_func(x_1)
lagrange_result_x_2 = lagrange_func(x_2)

print(f"Результат обчислення x = 2.25: {lagrange_result_x_1}")
print(f"Результат обчислення at x = 0.65: {lagrange_result_x_2}")

def divided_differences(x_data, y_data):
    n = len(y_data)
    coef = np.zeros([n, n])
    coef[:, 0] = y_data
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_data[i + j] - x_data[i])
    return coef[0, :]

def newton_polynomial(x_data, y_data, x):
    coeffs = divided_differences(x_data, y_data)
    n = len(coeffs)
    poly = coeffs[0]
    for i in range(1, n):
        term = coeffs[i]
        for j in range(i):
            term *= (x - x_data[j])
        poly += term
    return poly

newton_poly = newton_polynomial(x_values, L_n, x)

simplified_newton_poly = simplify(newton_poly)
print("\nПоліном Ньютона:")
print(simplified_newton_poly)

newton_func = lambdify(x, simplified_newton_poly, 'numpy')
newton_result_x_1 = newton_func(x_1)
newton_result_x_2 = newton_func(x_2)

print(f"Результат обчислення  x = 2.25: {newton_result_x_1}")
print(f"Результат обчислення  x = 0.65: {newton_result_x_2}")

print("Початкова функція ln^2(5x-2), x = 2.25:", np.log(5*2.25 - 2)**2)
print("Початкова функція  ln^2(5x-2), x = 0.65:",np.log(5*0.75 - 2)**2)

