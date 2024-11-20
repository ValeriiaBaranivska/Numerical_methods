import numpy as np
import pandas as pd
import plotly.express as px
from scipy.interpolate import CubicSpline
from sympy import symbols, simplify

a =  0
h_1 = 3
n = 10
x_i = [a]
x = symbols('x')

def f_x(x):
    return np.sin(x-4)

for i in range(1,n+1):
    x_k = a + h_1 * i
    x_i.append(x_k)
print(f"x_i: {x_i}")

h = []
for i in range(1,n+1):
    h_k = x_i[i] - x_i[i-1]
    h.append(h_k)

y_i = []
for i in x_i:
    y_i.append(f_x(i))
print(f"y_i: {y_i}")

#метод прогонки

def method_progonki(A, d):
    n = len(A)
    x = np.zeros(n)
    # Ініціалізація списків для коефіцієнтів
    a = np.zeros(n)  # нижня діагональ (під головною)
    b = np.zeros(n)  # головна діагональ
    c = np.zeros(n)  # верхня діагональ (над головною)

    for i in range(n):
        b[i] = A[i, i] # Головні коефіцієнти
        if i > 0:
            a[i] = A[i, i - 1] # Ліві коефіцієнти
        if i < n - 1:
            c[i] = A[i, i + 1] # Праві коефіцієнти


    print("Ліві коефіцієнти (a):", a)
    print("Головні коефіцієнти (b):", b)
    print("Праві коефіцієнти (c):", c)

    # Перевірка умови для методу прогонки
    for i in range(n):
        if abs(b[i]) < abs(a[i]) + abs(c[i]):
            raise ValueError(f"Матриця не задовольняє умови діагонального переважання в рядку {i}.")

    # прямий хід
    alpha_i = np.zeros(n)
    gamma_i = np.zeros(n)

    # обчислення прогоночних коефіцієнтів
    alpha_i[0] = -c[0] / b[0]
    gamma_i[0] = d[0] / b[0]
    for i in range(1, n):
        k = b[i] + a[i] * alpha_i[i - 1]
        if k == 0:
            raise ZeroDivisionError(f"Ділення на нуль у рядку {i}.")
        alpha_i[i] = -c[i] / k
        gamma_i[i] = (d[i] - a[i] * gamma_i[i - 1]) / k
        #print(k, alpha_i[i], gamma_i[i])
    # обернений хід
    x[-1] = gamma_i[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha_i[i] * x[i + 1] + gamma_i[i]
        print(x[i])

    # виведення округлених результатів
    print("Розв'язок системи:", np.round(x, 4))
    print("Прогоночні коефіцієнти Альфа:", np.round(alpha_i, 4))
    print("Прогоночні коефіцієнти Гамма:", np.round(gamma_i, 4))
    return x

# Заповнення матриці A та вектора F
A = np.zeros((n + 1, n + 1))
F = np.zeros(n + 1)

# граничні умови
A[0, 0] = 1
F[0] = 0

A[n, n] = 1
F[n] = 0

# F_i
for i in range(1, n):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    F[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])
# Виклик методу прогонки
c = method_progonki(A, F)
c_1 = np.linalg.solve(A,F)
print(f"c_1: {c_1}")
# Обчислення коефіцієнтів b та d
b = np.zeros(n)
d = np.zeros(n)
for i in range(n):
    b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])
a = y_i[:-1]

print(A)
print(F)

print(f"\nКоефіцієнти a:{a}")
print(f"Коефіцієнти b:{b}")
print(f"Коефіцієнти c:{c}")
print(f"Коефіцієнти d:{d}")

# Кількість точок для гладкого графіка
smooth_points = 500

# Масив для збереження значень x і y
x_smooth = []
y_smooth = []

# Обчислення значень сплайну на кожному інтервалі
for i in range(n):
    x_vals = np.linspace(x_i[i], x_i[i + 1], smooth_points // n)
    y_vals = [a[i] + b[i] * (x_val - x_i[i]) + c[i] * (x_val - x_i[i]) ** 2 + d[i] * (x_val - x_i[i]) ** 3 for x_val in
              x_vals]

    x_smooth.extend(x_vals)
    y_smooth.extend(y_vals)


# Функція для обчислення значень сплайну в точках x_i
def spline_func(x_val):
    for i in range(n):
        if x_i[i] <= x_val <= x_i[i + 1]:
            print(f"{x_i[i]} <= x <= {x_i[i + 1]} ---> {(a[i] + b[i] * (x_val - x_i[i]) + c[i] * (x_val - x_i[i])**2 + d[i] * (x_val - x_i[i])**3)} ")
            return (a[i] + b[i] * (x_val - x_i[i]) + c[i] * (x_val - x_i[i])**2 + d[i] * (x_val - x_i[i])**3)
    return None

# Функція для обчислення аналітичного вигляду кубічного сплайну
def analitics_spline():
    print("\n --- Кубічні сплайни:")
    for i in range(n):
        spline =  (a[i]
                    + b[i] * (x - x_i[i])
                    + c[i] * (x - x_i[i]) ** 2
                    + d[i] * (x - x_i[i]) ** 3)
        simp_spline = simplify(spline)
        print(f" #{i}   {simp_spline}")
analitics_spline()

spline_values = [spline_func(x) for x in x_i]
print(f"\n --- Значення сплайну в точках: {spline_values}")

# Перетворення в dataframe
df_nodes = pd.DataFrame({'x_i': x_i, 'y_i': y_i})
df_spline = pd.DataFrame({'x_i_1': x_smooth, 'y_i_1': y_smooth})

# побудова графіку
fig = px.line(df_nodes, x='x_i', y='y_i',
              title='Графік функції y = f(x) та кубічного сплайну',
              labels={'x_i': 'x', 'y_i': 'y'}
              )
fig.add_scatter(x=df_nodes['x_i'], y=df_nodes['y_i'], mode='lines', name='Початкова функція',
                line=dict(color='indigo'))

# Додаємо кубічний сплайн
fig.add_scatter(x=df_spline['x_i_1'], y=df_spline['y_i_1'], mode='lines', name='Кубічний сплайн',
                line=dict(color='green'), marker=dict(symbol='x', size=10, color='green'))

# Додаємо точки початкової функції
fig.add_scatter(x=df_nodes['x_i'], y=df_nodes['y_i'], mode='markers', name='Точки початкової функції',
                marker=dict(symbol='circle', size=10, color='purple'))
fig.show()

#перевірка вбудованими бібліотеками
x_i_1 = np.arange(0,31, h_1)  # вибір діапазону для x
y_i_1 = f_x(x_i_1)

# Створюємо кубічний сплайн
cs = CubicSpline(x_i_1, y_i_1, bc_type='natural')  # bc_type="natural" для природного сплайну

# Обчислення значень сплайну для щільної сітки
x_dense = np.linspace(min(x_i_1), max(x_i_1), 500)  # Щільна сітка для гладкого графіка
y_dense = cs(x_dense)

df = pd.DataFrame({
    'x_i_1': x_i_1,
    'y_i_1': y_i_1,
    'spline_values': cs(x_i_1)  # Обчислення значень сплайну в точках x_i_1
})
# Побудова графіку
fig = px.line(df, x='x_i_1', y='y_i_1',
              title="Кубічний сплайн через вбудовану бібліотеку  ",
              labels={'x_i_1': 'x', 'y_i_1': 'y'})

fig.add_scatter(x=df_nodes['x_i'], y=df_nodes['y_i'], mode='lines', name='Початкова функція',
                line=dict(color='indigo'))
# Додаємо кубічний сплайн
fig.add_scatter(x=x_dense, y=y_dense, mode='lines', name='Кубічний сплайн',
                line=dict(color='green'))

# Додаємо точки початкової функції
fig.add_scatter(x=df['x_i_1'], y=df['y_i_1'], mode='markers', name='Точки початкової функції',
                marker=dict(symbol='circle', size=8, color='purple'))
fig.show()


