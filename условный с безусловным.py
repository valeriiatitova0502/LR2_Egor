import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Определяем входные параметры
a = 4
b = 1
c = 1
d = 2
f_coef = -2
e = -15
x0 = np.array([9, 1])
tol = 0.01
R_values = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]

# Определяем целевую функцию с обратным штрафом
def f(x, R=None):
    x1, x2 = x
    base_value = (x1 - a) ** 2 + (x2 - b) ** 2 + c * x1 * x2
    if R is not None:
        g_x = d * x1 + f_coef * x2 + e
        if g_x <= 0:
            g_x = min_penalty  # Избегаем деления на ноль или отрицательных значений
        penalty = R / g_x
        return base_value + penalty
    else:
        return base_value

# Определяем градиент целевой функции
def grad_f(x, R=None):
    x1, x2 = x
    df_dx1 = 2 * (x1 - a) + x2
    df_dx2 = 2 * (x2 - b) + x1
    if R is not None:
        g_x = d * x1 + f_coef * x2 + e
        if g_x <= 0:
            g_x = min_penalty  # Избегаем деления на ноль или отрицательных значений
        penalty_grad_x1 = -R * d / (g_x ** 2)
        penalty_grad_x2 = -R * f_coef / (g_x ** 2)
        df_dx1 += penalty_grad_x1
        df_dx2 += penalty_grad_x2
    return np.array([df_dx1, df_dx2])
min_penalty = 1e-8

# Функция для минимизации методом Сильвестра
def minimize_sylvester(x0, f, grad_f, R):
    res = minimize(f, x0, args=(R,), method='L-BFGS-B', jac=grad_f,
                   options={'ftol': tol, 'maxiter': 1000, 'disp': False})
    if res.success:
        return res.x
    else:
        return None

# Функция для метода сопряженных градиентов с фиксированным шагом
def cg_method(f, grad_f, x0, R_values, tol=1e-6, max_iter=2000, step_size=0.005):
    optimums = []
    for R in R_values:
        x = x0.astype(float)
        for iteration in range(max_iter):
            grad = grad_f(x, R)
            x -= step_size * grad
            if np.linalg.norm(grad) < tol:
                break
        optimums.append(x)
    return np.array(optimums)

# Функция для оптимизации и построения графика
def optimize_and_plot(x0):
    x1_values = np.linspace(-15, 15, 400)
    x2_values = np.linspace(-14, 20, 400)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    Z = np.zeros_like(X1)
    for i in range(len(x1_values)):
        for j in range(len(x2_values)):
            Z[j, i] = f([X1[j, i], X2[j, i]], R_values[0])

    plt.figure(figsize=(10, 8))

    # Вызываем метод сопряженных градиентов
    optimums = cg_method(f, grad_f, x0, R_values)

    num_contours = 4
    contour_radius = 45
    for i in range(2):
        current_point = x0 if i == 0 else optimums[i - 1]
        x1_min = current_point[0] - contour_radius
        x1_max = current_point[0] + contour_radius
        x2_min = current_point[1] - contour_radius
        x2_max = current_point[1] + contour_radius
        x1 = np.linspace(x1_min, x1_max, 100)
        x2 = np.linspace(x2_min, x2_max, 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)
        for j in range(len(x1)):
            for k in range(len(x2)):
                Z[k, j] = f([X1[k, j], X2[k, j]])

        optimal_value = f(current_point)
        contour_levels = np.linspace(optimal_value, optimal_value + 2 * contour_radius, num_contours)
        plt.contour(X1, X2, Z, levels=contour_levels, colors='gray', linestyles='solid', linewidths=1, extend='both')

    # Начальная точка как красная линия
    plt.plot([x0[0]], [x0[1]], 'r-', linewidth=2)

    for i in range(len(optimums) - 1):
        plt.plot([optimums[i][0], optimums[i + 1][0]], [optimums[i][1], optimums[i + 1][1]], color='red')
        plt.plot(optimums[i][0], optimums[i][1], 'ro')
    plt.plot([x0[0], optimums[0][0]], [x0[1], optimums[0][1]], color='red')

    plt.contour(X1, X2, d * X1 + f_coef * X2 + e, levels=[0], colors='blue')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()

    # Выводим результаты оптимизации
    print("Входные параметры:", "a =", a, "b =", b, "c =", c, "d =", d, "f =", f_coef, "e =", e, "x0 =", x0, "tol =", tol)
    for i, R in enumerate(R_values):
        print(f"Значения оптимума для R={R} методом сопряженных градиентов с штрафом обратной функции:", optimums[i])
        sylvester_optimal_point = minimize_sylvester(x0, f, grad_f, R)
        if sylvester_optimal_point is not None:
            print(f"Проверка методом Сильвестра найдена точка оптимума для R={R} при (x1, x2) =", sylvester_optimal_point)
        else:
            print(f"Проверить методом Сильвестра для R={R} не удалось найти точку оптимума.")

# Вызываем функцию для оптимизации и построения графика
optimize_and_plot(x0)

