import time
import tkinter as tk
from tkinter import messagebox, StringVar

import numpy as np
import optuna
import scipy
import sympy
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import random
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)


def reset():
    global start_point_labels, start_point_entries, function, dimension, v, function_inf, dimension_inf, \
        x_left, x_right, x_step, y_left, y_right, y_step, func, contour
    while len(contour) > 0:
        contour[-1].remove()
        contour = []
    old_dimension = dimension
    dimension = dimension_entry.get()

    if not dimension.isdigit():
        messagebox.showerror('Python Error', "Expected integer >= 1 in dimension")
        return
    dimension = int(dimension)
    if dimension <= 0:
        messagebox.showerror('Python Error', "Expected integer >= 1 in dimension")

    v = sympy.symbols(dimension_vars())

    try:
        func = parse_expr(function_entry.get())
        function = lambdify(v, func)
    except ValueError:
        messagebox.showerror("Input Error", "Invalid Function")

    x_left_label.grid_remove()
    x_left_entry.grid_remove()
    x_right_label.grid_remove()
    x_right_entry.grid_remove()
    x_step_label.grid_remove()
    x_step_entry.grid_remove()
    y_left_label.grid_remove()
    y_left_entry.grid_remove()
    y_right_label.grid_remove()
    y_right_entry.grid_remove()
    y_step_label.grid_remove()
    y_step_entry.grid_remove()

    try:
        if 0 < dimension < 3:
            x_left = float(x_left_entry.get())
            x_right = float(x_right_entry.get())
            x_step = float(x_step_entry.get())
            x_left_label.grid(row=2, column=0, sticky='W')
            x_left_entry.grid(row=2, column=1, sticky='W')
            x_right_label.grid(row=2, column=2, sticky='W')
            x_right_entry.grid(row=2, column=3, sticky='W')
            x_step_label.grid(row=2, column=4, sticky='W')
            x_step_entry.grid(row=2, column=5, sticky='W')
            if dimension > 1:
                y_left = float(y_left_entry.get())
                y_right = float(y_right_entry.get())
                y_step = float(y_step_entry.get())
                y_left_label.grid(row=3, column=0, sticky='W')
                y_left_entry.grid(row=3, column=1, sticky='W')
                y_right_label.grid(row=3, column=2, sticky='W')
                y_right_entry.grid(row=3, column=3, sticky='W')
                y_step_label.grid(row=3, column=4, sticky='W')
                y_step_entry.grid(row=3, column=5, sticky='W')

    except ValueError:
        messagebox.showerror("Value Error", "Incorrect position inputs")
        return

    try:
        build_graphic()
    except TypeError:
        messagebox.showerror("Input Error",
                             "Invalid variables in the function input field for {0}-dimensional format".format(
                                 dimension))
        dimension = old_dimension
        return
    function_inf.set(str(func))
    dimension_inf.set(dimension_info())

    for elem in start_point_labels:
        elem.destroy()
    if dimension == 1:
        args = ["x"]
    elif dimension == 2:
        args = ["x", "y"]
    else:
        args = ["x" + str(i) for i in range(1, dimension + 1)]
    start_point_labels = [tk.Label(root, text="Начальная координата {0}:".format(arg)) for arg in args]

    if dimension >= old_dimension:
        start_point_entries = start_point_entries + [tk.Entry(root) for _ in range(dimension - old_dimension)]
    else:
        for i in range(dimension, old_dimension):
            start_point_entries[i].destroy()
        start_point_entries = start_point_entries[:dimension]
    for i in range(old_dimension, dimension):
        start_point_entries[i].insert(0, "1")

    for i in range(dimension):
        start_point_labels[i].grid(row=30 + i, column=0, columnspan=2, sticky='W')
    for i in range(old_dimension, dimension):
        start_point_entries[i].grid(row=30 + i, column=2, sticky='W')
    start_button.grid(row=30 + dimension, columnspan=6, pady=4)


def on_method_select(_):
    global settings
    method = method_entry.get()
    if method is None:
        return
    for dict_key, dict_value in settings.items():
        for elements in dict_value:
            for element in elements:
                element.grid_remove()
    labels, entries = settings.get(method)
    for i in range(len(labels)):
        labels[i].grid(row=9 + i, column=0, sticky='W')
        entries[i].grid(row=9 + i, column=1, sticky='W')


def dimension_info():
    global dimension
    return str(dimension) + ", vars: " + dimension_vars(full=False)


def dimension_vars(full=True):
    global dimension
    match dimension:
        case 1:
            return "x"
        case 2:
            return "x y"
        case _:
            if full:
                return " ".join(["x" + str(i) for i in range(1, dimension + 1)])
            else:
                return "x1 x2 ..."


def nelder_mead():
    global points, start_point, values
    points = scipy.optimize.minimize(lambda x: function(*x), start_point, method="nelder-mead",
                                     options={'return_all': True}).get("allvecs")
    values = [function(*p) for p in points]
    return


def start():
    global methods, start_point, counts

    try:
        start_point = [float(entry.get()) for entry in start_point_entries]
    except ValueError:
        messagebox.showerror('Value Error', "Invalid start point")
        return

    counts = 0
    method = method_entry.get()

    if method == methods[0]:
        annealing()
    elif method == methods[1]:
        gradient_descent()
    elif method == methods[2]:
        nelder_mead()
    elif method == methods[3]:
        optuna_optimize()
    else:
        messagebox.showerror("Input error", "Unknown method")
        return

    iterations.set(str(len(points)))
    counts_of_function.set(str(counts))

    add_contour()
    print_result(points[-1], values[-1])


# start point is useless
def optuna_optimize():
    global points, values, function, start_point, counts

    try:
        lb = float(left_border_entry.get())
    except ValueError:
        messagebox.showerror('Value Error', "Invalid left border")
        return
    try:
        rb = float(right_border_entry.get())
    except ValueError:
        messagebox.showerror('Value Error', "Invalid right border")
        return
    try:
        trials = int(trials_entry.get())
    except ValueError:
        messagebox.showerror('Value Error', "Invalid numer of trials")
        return

    study_object = object_entry.get()

    def objective(trial):
        return function(*[trial.suggest_float(i, lb, rb) for i in v])

    def objective_gd(trial):
        step_entry.delete(0, tk.END)
        trial.suggest_float("step", lb, rb)
        gradient_descent()
        if distance(points[-1], np.array([0] * dimension) < float(epsilon_entry.get())):
            return len(points)
        else:
            return 10000  # не нашел
        
    if study_object == "функция":

        study = optuna.create_study()
        study.optimize(objective, n_trials=trials)

        points = []
        values = []
        for tr in study.trials:
            points.append(np.array([tr.params[i] for i in v]))
            values.append(tr.value)
    else:
        def objective_gd(trial):
            step_entry.delete(0, tk.END)
            trial.suggest_float("step", lb, rb)
            gradient_descent()
            if distance(points[-1], np.array([0] * dimension) < float(epsilon_entry.get())) :
                return len(points)
            else:
                return 10000  # не нашел


    res = np.array(list(study.best_params.values()))
    points.append(res)
    values.append(function(*res))


def annealing():
    global points, values, function, start_point, counts

    try:
        temp_step = float(temp_entry.get())
        if temp_step <= 0:
            messagebox.showerror("Value Error", "Temperature step must be positive and not zero")
    except ValueError:
        messagebox.showerror('Value Error', "Invalid temperature")
        return

    try:
        step = float(step_entry.get())
        if step <= 0:
            messagebox.showerror("Value Error", "Step must be positive and not zero")
    except ValueError:
        messagebox.showerror('Value Error', "Invalid temperature")
        return

    temp = 1
    points = [np.array(start_point)]
    while temp > 0:
        try:
            next_point = new_point_rand(points[-1], step)
        except RuntimeWarning:
            break
        if function(*next_point) < function(*points[-1]):
            points.append(next_point)
        elif random.random() < temp:
            points.append(next_point)
        counts += 1
        temp -= temp_step
    values = np.array([function(*start_point) for start_point in points])


def distance(p1, p2):
    return np.sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(p1.size)]))


def gradient_descent():
    global points, values, start_point, counts
    try:
        epsilon = float(epsilon_entry.get())
        if epsilon <= 0:
            messagebox.showerror("Value Error", "Epsilon must be positive and not zero")
    except ValueError:
        messagebox.showerror('Value Error', "Invalid temperature")
        return

    try:
        step = float(step_entry.get())
        if step <= 0:
            messagebox.showerror("Value Error", "Step must be positive and not zero")
    except ValueError:
        messagebox.showerror('Value Error', "Invalid temperature")
        return

    try:
        grad = df_point()
    except SyntaxError:
        messagebox.showerror('Counting Error', "Can not differentiate the function")
        return

    try:
        dim1_method = dim1_entry.get()
    except SyntaxError:
        messagebox.showerror('Counting Error', "Can not differentiate the function")
        return

    timelimit = 20
    start_time = time.time()
    points = [np.array(start_point)]
    values = [function(*points[-1])]
    while time.time() - start_time < timelimit:
        if any([abs(x) > LIMIT for x in points[-1]]):
            break
        shift = np.array([f(*points[-1]) * step for f in grad])
        counts += dimension
        if dim1_method == dim1_methods[0]:
            points.append(points[-1] - shift)
        elif dim1_method == dim1_methods[1]:
            points.append(fib_search(points[-1] - shift, points[-1] + shift, epsilon))
        values.append(function(*points[-1]))
        if distance(points[-1], points[-2]) < epsilon:
            break


def fib_search(a, b, epsilon):
    global function, counts
    n = len(a)
    golden_ratio = scipy.constants.golden_ratio
    p1 = [0] * n
    p2 = [0] * n
    for i in range(n):
        p1[i] = b[i] - (b[i] - a[i]) / golden_ratio
        p2[i] = a[i] + (b[i] - a[i]) / golden_ratio
    y1 = function(*p1)
    y2 = function(*p2)
    counts += 2
    while distance(a, b) > epsilon:
        if y1 >= y2:
            for i in range(n):
                a[i] = p1[i]
                p1[i] = p2[i]
            for i in range(n):
                p2[i] = a[i] + (b[i] - a[i]) / golden_ratio
            y1 = y2
            y2 = function(*p2)
        else:
            for i in range(n):
                b[i] = p2[i]
                p2[i] = p1[i]
            for i in range(n):
                p1[i] = b[i] - (b[i] - a[i]) / golden_ratio
            y2 = y1
            y1 = function(*p1)
        counts += 1
    return np.array([(a[i] + b[i]) / 2 for i in range(n)])


def print_result(point, func_value):
    global dimension
    match dimension:
        case 1, 2:
            result.set("(" + ", ".join(["{:.8f}".format(coord) for coord in point]) + ")")
        case _:
            result.set("(" + ",\n".join(["{:.2f}".format(coord) for coord in point]) + ")")
    value.set("{:.6f}".format(func_value))


def new_temp(temp, temp_step):
    return temp - temp_step


# Случайное блуждание, шаг по каждой координате < step
def new_point_rand(point, step):
    return [x + random.uniform(-step, step) for x in point]


# Блуждание в направление градиента или анти градиента, размер шага < step
def new_point_grad(point, grad, step):
    grad_step = random.uniform(-step, step)
    return point - np.array([f(*point) * grad_step for f in grad])


def df_point():
    global function, v
    if dimension == 1:
        return [lambdify(v, e) for e in np.array([sympy.diff(func)])]
    return [lambdify(v, e) for e in np.array([sympy.diff(func, i) for i in v])]


def add_contour():
    global points, values
    global dimension, graphics, contour
    if len(contour) > 0:
        contour[-1].remove()
        contour = []
    if len(points) == 0:
        return
    np_points = np.array(points)
    match dimension:
        case 1:
            contour = plt.plot(np_points[:, 0], values, 'o-', c="#FF0000")
        case 2:
            contour = plt.plot(np_points[:, 0], np_points[:, 1], values, 'o-', c="#FF0000")
        case _:
            return
    graphics.draw()


# График
def build_graphic():
    global x_step, x_left, x_right, y_step, y_left, y_right
    global function, graphics, dimension
    fig = plt.figure()
    match dimension:
        case 1:
            x_grid = np.arange(x_left, x_right, x_step)
            y_grid = function(x_grid)
            ax = fig.add_subplot(121)
            ax.plot(x_grid, y_grid, alpha=0.2)
        case 2:
            x_grid, y_grid = np.meshgrid(np.arange(x_left, x_right, x_step), np.arange(y_left, y_right, y_step))
            z_grid = function(x_grid, y_grid)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(x_grid, y_grid, z_grid, alpha=0.2)
        case _:
            pass
    if len(contour) > 0:
        add_contour()
    graphics.get_tk_widget().destroy()
    graphics = FigureCanvasTkAgg(fig, master=root)
    graphics.draw()
    graphics.get_tk_widget().grid(row=100, columnspan=100, sticky='W')


# Создание окна
root = tk.Tk()
root.title("Метод отжига")
tk.font.nametofont("TkDefaultFont").configure(family="Times New Roman", size=16, weight=tk.font.BOLD)
tk.font.nametofont("TkFixedFont").configure(family="JetBrains Mono", size=16, weight=tk.font.BOLD)
tk.font.nametofont("TkTextFont").configure(family="JetBrains Mono", size=16, weight=tk.font.BOLD)

# Данные
LIMIT = 10e10  # limit for a search
functions = ["x**2 + y**2",
             "100*x**2 + y**2",
             "0.01*x**2 + y**2",
             "(x**2 + y**2) * ((x-1)**2 + (y-1)**2)",
             "(1 - x) ** 2 + 100 * (y - x ** 2) ** 2"]
methods = ["отжиг с постоянным шагом температуры",
           "градиентный спуск",
           "Нелдер-мид (scipy)",
           "Optuna"]
dim1_methods = ["Нет", "Метод золотого сечения"]
function = parse_expr("x ** 2 + y ** 2")
func = parse_expr("x ** 2 + y ** 2")
dimension = 2
v = sympy.symbols("x y")
graphics = FigureCanvasTkAgg(master=root)
contour = plt.plot()
start_point = np.array([0])
counts = 0  # Количество вычислений значений функции
result = StringVar()
value = StringVar()
iterations = StringVar()
counts_of_function = StringVar()
dimension_inf = StringVar()
dimension_inf.set(dimension_info())
function_inf = StringVar()
function_inf.set(function)
points = []
values = []

x_left = -2
x_right = 2
x_step = 0.1
y_left = -2
y_right = 2
y_step = 0.1

# Название для полей
function_label = tk.Label(root, text="Функция:")
function_current = tk.Entry(root, textvariable=function_inf, state="readonly", font="TkFixedFont", background="#99cccc",
                            relief="groove")

dimension_label = tk.Label(root, text="Измерение")
dimension_current = tk.Entry(root, textvariable=dimension_inf, state="readonly", font="TkFixedFont",
                             background="#99cccc", relief="groove")

x_left_label = tk.Label(root, text="Левая граница x")
x_right_label = tk.Label(root, text="Правая граница x")
x_step_label = tk.Label(root, text="Шаг по x")

y_left_label = tk.Label(root, text="Левая граница y")
y_right_label = tk.Label(root, text="Правая граница y")
y_step_label = tk.Label(root, text="Шаг по y")

method_label = tk.Label(root, text="Метод:")

result_label = tk.Label(root, text="Результат:")
value_label = tk.Label(root, text="Значение:")
iterations_label = tk.Label(root, text="Число итераций:")
counts_label = tk.Label(root, text="Количество вычислений функции:")

start_point_labels = []

# Поля ввода

function_entry = ttk.Combobox(root, width=40, values=functions)
function_entry.insert(0, str(function))

dimension_entry = tk.Entry(root)
dimension_entry.insert(0, str(dimension))

start_point_entries = [tk.Entry(root) for _ in range(dimension)]
for el in start_point_entries:
    el.insert(0, "1")

x_left_entry = tk.Entry(root, width=5)
x_right_entry = tk.Entry(root, width=5)
x_step_entry = tk.Entry(root, width=5)
y_left_entry = tk.Entry(root, width=5)
y_right_entry = tk.Entry(root, width=5)
y_step_entry = tk.Entry(root, width=5)

x_left_entry.insert(0, str(x_left))
x_right_entry.insert(0, str(x_right))
x_step_entry.insert(0, str(x_step))
y_left_entry.insert(0, str(y_left))
y_right_entry.insert(0, str(y_right))
y_step_entry.insert(0, str(y_step))

result_entry = tk.Entry(root, textvariable=result, fg="black", bg="white", bd=0, state="readonly", width=50)
value_entry = tk.Entry(root, textvariable=value, fg="black", bg="white", bd=0, state="readonly", width=50)
iterations_entry = tk.Entry(root, textvariable=iterations, fg="black", bg="white", bd=0, state="readonly", width=50)
counts_entry = tk.Entry(root, textvariable=counts_of_function, fg="black", bg="white", bd=0, state="readonly", width=50)

method_entry = ttk.Combobox(root, values=methods, width=40, state="readonly")
method_entry.bind("<<ComboboxSelected>>", on_method_select)

# Настройки
temp_label = tk.Label(root, text="Шаг температуры:")
step_label = tk.Label(root, text="Шаг точки")
epsilon_label = tk.Label(root, text="Условие останова: шаг < эпсилон")
dim1_label = tk.Label(root, text="Метод одномерного поиска")
left_border_label = tk.Label(root, text="Левая граница поиска")
right_border_label = tk.Label(root, text="Правая граница поиска")
trials_label = tk.Label(root, text="Количество попыток")
object_label = tk.Label(root, text="Объект изучения")

annealing_labels = [step_label, temp_label]
gradient_descent_labels = [step_label, epsilon_label, dim1_label]
optuna_labels = [left_border_label, right_border_label, trials_label, object_label]

step_entry = tk.Entry(root)
step_entry.insert(0, "0.1")

temp_entry = tk.Entry(root)
temp_entry.insert(0, "0.01")

epsilon_entry = tk.Entry(root)
epsilon_entry.insert(0, "0.001")

dim1_entry = ttk.Combobox(root, values=dim1_methods, width=40, state="readonly")
dim1_entry.set(dim1_methods[0])

left_border_entry = tk.Entry(root)
right_border_entry = tk.Entry(root)
trials_entry = tk.Entry(root)
left_border_entry.insert(0, "-2")
right_border_entry.insert(0, "2")
trials_entry.insert(0, "100")
object_entry = ttk.Combobox(root, values=["функция", "гиперпараметры"], width=40, state="readonly")
object_entry.set("функция")

annealing_entries = [step_entry, temp_entry]
gradient_descent_entries = [step_entry, epsilon_entry, dim1_entry]
optuna_entries = [left_border_entry, right_border_entry, trials_entry]

settings = {methods[0]: (annealing_labels, annealing_entries),
            methods[1]: (gradient_descent_labels, gradient_descent_entries),
            methods[2]: ([], []),
            methods[3]: (optuna_labels, optuna_entries)}

# Кнопки
start_button = tk.Button(root, text="Запустить", command=start)
reset_button = tk.Button(root, text="Применить", command=reset)

# Размещение элементов в окне
function_label.grid(row=0, column=0, sticky='W')
function_entry.grid(row=0, column=1, columnspan=2, sticky='W')
function_current.grid(row=0, column=2, columnspan=4, sticky='W')

dimension_label.grid(row=1, column=0, sticky='W')
dimension_entry.grid(row=1, column=1, sticky='W')
dimension_current.grid(row=1, column=2, columnspan=4, sticky='W')

reset_button.grid(row=6, columnspan=3, pady=4)

method_label.grid(row=7, column=0, sticky='W')
method_entry.grid(row=7, column=1, columnspan=4, sticky='W')

result_label.grid(row=21, column=0, sticky='W')
result_entry.grid(row=21, column=1, columnspan=6, sticky='W')
value_label.grid(row=22, column=0, sticky='W')
value_entry.grid(row=22, column=1, columnspan=6, sticky='W')
iterations_label.grid(row=23, column=0, sticky='W')
iterations_entry.grid(row=23, column=1, columnspan=6, sticky='W')
counts_label.grid(row=24, column=0, sticky='W')
counts_entry.grid(row=24, column=1, columnspan=6, sticky='W')

for j in range(dimension):
    start_point_entries[j].grid(row=30 + j, column=2, sticky='W')

graphics.get_tk_widget().grid(row=100, columnspan=100, sticky='W')

# Запуск главного цикла
reset()
root.mainloop()
