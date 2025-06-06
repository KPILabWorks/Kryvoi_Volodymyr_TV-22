import matplotlib.pyplot as plt
import numpy as np

sleeping_zone = [38, 40, 42, 39, 41, 43, 44, 40, 41, 39, 60, 41, 43, 38, 40, 39, 41, 77, 39, 40]
central_zone = [52, 55, 58, 60, 56, 59, 57, 61, 62, 59, 70, 63, 66, 68, 65, 60, 58, 73, 62, 61]

minutes = list(range(1, len(sleeping_zone) + 1))

avg_sleeping = np.mean(sleeping_zone)
avg_central = np.mean(central_zone)

plt.figure(figsize=(12, 5))
plt.plot(minutes, sleeping_zone, label='Спальна зона', marker='o')
plt.plot(minutes, central_zone, label='Центральна вулиця', marker='o')
plt.axhline(y=avg_sleeping, color='blue', linestyle='--', label=f'Середній (спальна): {avg_sleeping:.1f} дБ')
plt.axhline(y=avg_central, color='orange', linestyle='--', label=f'Середній (центр): {avg_central:.1f} дБ')

plt.title('Рівень шуму вночі у Вишневому')
plt.xlabel('Хвилина вимірювання')
plt.ylabel('Рівень шуму (дБ)')
plt.xticks(minutes)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
