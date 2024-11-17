import numpy as np
import pandas as pd
import plotly.express as px
from scipy.interpolate import CubicSpline
from sympy import symbols, simplify, lambdify

a =  1
h = 1
x_i = [a]
x = symbols('x')

for i in range(1,11):
    x_k = a + h * i
    x_i.append(x_k)
print(f"x_i: {x_i}")

n = len(x_i) - 1

h = []
for i in range(1,len(x_i)):
    h_k = x_i[i] - x_i[i-1]
    h.append(h_k)

y_i = []
for i in x_i:
    y_i.append(round(np.log(5*i -2)**2,8))
print(f"y_i: {y_i}")

#метод прогонки
A = np.zeros((n + 1, n + 1))
F = np.zeros(n + 1)

A[0, 0] = 1  # c_0 = 0
A[n, n] = 1  # c_n = 0
#F_i
for i in range(1, n):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    F[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])

c = np.linalg.solve(A, F) #розвязання системи рівнянь

b = np.zeros(n)
d = np.zeros(n)
for i in range(n):
    b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])
a = y_i[:-1]

#рівняння сплайну f_i(x) = a_i + b_i(x-x_i) +c_i (x-x_i)**2+d_i(x-x_i)**3
# Функція для обчислення значень сплайну в точках x_i
def spline_func(x_val):
    for i in range(n):
        if x_i[i] <= x_val <= x_i[i + 1]:
            return (a[i] + b[i] * (x_val - x_i[i]) + c[i] * (x_val - x_i[i])**2 + d[i] * (x_val - x_i[i])**3)
    return None

def analitics_spline():
    for i in range(n):
        spline = + (a[i]
                    + b[i] * (x - x_i[i])
                    + c[i] * (x - x_i[i]) ** 2
                    + d[i] * (x - x_i[i]) ** 3)

    simplified_spline = simplify(spline)
    print(f"Кубічний сплайн: {simplified_spline}")
analitics_spline()

spline_values = [spline_func(x) for x in x_i]
print(f"Значення сплайну в точках: {spline_values}")

# Перетворення в dataframe
df = pd.DataFrame({'x_i': x_i, 'y_i': y_i,
                   'spline_values': spline_values})

# побудова графіку
fig = px.line(df, x='x_i', y='y_i',
              title='Графік функції y = f(x) та кубічного сплайну',
              labels={'x_i': 'x', 'y_i': 'y'},
              markers=True)

# Додаємо кубічний сплайн
fig.add_scatter(x=df['x_i'], y=df['spline_values'], mode='lines', name='Кубічний сплайн',
                line=dict(color='green'), marker=dict(symbol='x', size=10, color='green'))

# Додаємо точки початкової функції
fig.add_scatter(x=df['x_i'], y=df['y_i'], mode='markers', name='Точки початкової функції',
                marker=dict(symbol='circle', size=8, color='purple'))
fig.show()

#перевірка вбудованими бібліотеками

x_i_1 = np.linspace(1, 12, 11)  # вибір діапазону для x
y_i_1 = np.log(5*x_i_1 -2)**2

# Створюємо кубічний сплайн
cs = CubicSpline(x_i_1, y_i_1, bc_type='natural')  # bc_type="natural" для природного сплайну

# Обчислення значень сплайну для щільної сітки x_dense
x_dense = np.linspace(min(x_i_1), max(x_i_1), 500)  # Щільна сітка для гладкого графіка
y_dense = cs(x_dense)

df = pd.DataFrame({
    'x_i_1': x_i_1,
    'y_i_1': y_i_1,
    'spline_values': cs(x_i_1)  # Обчислення значень сплайну в точках x_i_1
})
# Побудова графіку
fig = px.line(df, x='x_i_1', y='y_i_1',
              title=r"Кубічний сплайн для $ln^2(5x - 2)$",
              labels={'x_i_1': 'x', 'y_i_1': 'y'},
              markers=True)

# Додаємо кубічний сплайн
fig.add_scatter(x=x_dense, y=y_dense, mode='lines', name='Кубічний сплайн',
                line=dict(color='green'))

# Додаємо точки початкової функції
fig.add_scatter(x=df['x_i_1'], y=df['y_i_1'], mode='markers', name='Точки початкової функції',
                marker=dict(symbol='circle', size=8, color='purple'))
fig.show()


