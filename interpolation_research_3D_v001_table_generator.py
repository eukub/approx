import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Можно использовать другой бэкэнд, если необходимо

# Генерация данных
x_values = np.arange(-3, 3.24, 0.25)
y_values = np.arange(0, 25.1, 5)

# Создание сетки значений x и y
X, Y = np.meshgrid(x_values, y_values)
# F = X**2 * Y**2 + 2
# F = X**2 + Y**2 + 2
F = np.sin(X) * np.cos(Y)

# Создание таблицы значений и сохранение в CSV-файл
data = {'x': X.flatten(), 'y': Y.flatten(), 'f(x, y)': F.flatten()}
df = pd.DataFrame(data)
df.to_csv('data_table.csv', index=False)

# Построение трехмерного графика
fig = plt.figure(figsize=(2 * plt.figaspect(1)))
ax = fig.add_subplot(111, projection='3d')

# Отображение поверхности
ax.plot_surface(X, Y, F, cmap='viridis', alpha=0.8)

# Подписи осей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')

# Показать график
plt.show()
