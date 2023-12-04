import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Можно использовать другой бэкэнд, если необходимо

POWER_OF_APPROXIMATION_POLYNOM = 16

df = pd.read_csv('data_table.csv')
X = df['x'].values
Y = df['y'].values
F = df['f(x, y)'].values

vals_dict = {(X[idx], Y[idx]): item for idx, item in enumerate(F)}

# Вычисление размеров массивов
num_x_values = len(np.unique(X))
num_y_values = len(np.unique(Y))

len_of_x_section = X[1] - X[0]
len_of_y_section = Y[num_x_values] - Y[0]

# Построение трехмерного графика
fig = plt.figure(figsize=(18, 9))
ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_trisurf(X, Y, F, cmap='viridis', alpha=0.8)
ax1.scatter(X, Y, F, color='r', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('F(X, Y)')
ax1.set_title('Original F(X, Y)')
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


left_x_left_bound = -6
left_x_right_bound = 6
num_of_new_x = int((X[-1] - left_x_left_bound + len_of_x_section) / len_of_x_section)
X_marks = np.linspace(
    start=left_x_left_bound,
    stop=left_x_right_bound + len_of_x_section,
    num=num_of_new_x,
    endpoint=False
)
X_appr = np.zeros(0)
for y_idx in range(num_y_values):
    X_tmp = np.concatenate((X_appr.reshape(1, -1), X_marks.reshape(1, -1)), axis=1)
    X_appr = X_tmp
X_appr = X_appr[0]    # NOTE: Ready

left_y_left_bound = 0
num_of_new_y = int((Y[-1] - left_y_left_bound + len_of_y_section) / len_of_y_section)
Y_appr = np.zeros(0)
for y_idx, y_val in enumerate(np.unique(Y)):
    Y_marks = np.full(num_of_new_x, y_val)
    Y_tmp = np.concatenate((Y_appr.reshape(1, -1), Y_marks.reshape(1, -1)), axis=1)
    Y_appr = Y_tmp
Y_appr = Y_appr[0]


F_appr = np.zeros(X_appr.size)
for y_idx, y_val in enumerate(np.unique(Y)):
    fn_model = np.poly1d(
        np.polyfit(
            X[num_x_values*y_idx: num_x_values*(y_idx + 1)],
            F[num_x_values*y_idx: num_x_values*(y_idx + 1)],
            POWER_OF_APPROXIMATION_POLYNOM
        )
    )
    for x_val_idx, x_val in enumerate(X_appr[num_of_new_x*y_idx:num_of_new_x*(y_idx+1)]):
        if not vals_dict.get((x_val, y_val), False):
            F_appr[y_idx * num_of_new_x + x_val_idx] = fn_model(x_val)
        else:
            F_appr[y_idx * num_of_new_x + x_val_idx] = vals_dict[(x_val, y_val)]

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_appr, Y_appr, F_appr, color='green', marker='.')
ax2.set_xlabel('X_appr')
ax2.set_ylabel('Y_appr')
ax2.set_zlabel('F_appr(x, y)')
ax2.set_title('Approximation chart | F_appr(X_appr, Y_appr)')
ax2.legend()

plt.show()
