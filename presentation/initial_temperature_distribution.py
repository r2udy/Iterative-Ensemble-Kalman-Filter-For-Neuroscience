import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi, 500)
u = np.sin(x) + 0.4*np.sin(3*x) + 0.12*np.sin(5*x) + 0.04*np.sin(7*x)

plt.figure(figsize=(8, 4))
plt.plot(x, u, color='steelblue', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x, 0)')
plt.title('Initial temperature distribution')
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
           ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.axhline(0, color='gray', linewidth=0.7, linestyle='--')
plt.tight_layout()
plt.show()


x = np.linspace(0, np.pi, 500)
alpha = 0.08
times = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

def u(x, t):
    return (    1.00 * np.sin(1*x) * np.exp(-alpha * 1**2 * t)
              + 0.40 * np.sin(3*x) * np.exp(-alpha * 3**2 * t)
              + 0.12 * np.sin(5*x) * np.exp(-alpha * 5**2 * t)
              + 0.04 * np.sin(7*x) * np.exp(-alpha * 7**2 * t))

plt.figure(figsize=(9, 5))
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.1, 0.85, len(times)))

for t, c in zip(times, colors):
    plt.plot(x, u(x, t), color=c, linewidth=2, label=f't = {t}')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Heat equation — temporal evolution')
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
           ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.axhline(0, color='gray', linewidth=0.7, linestyle='--')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()