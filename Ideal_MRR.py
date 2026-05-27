import numpy as np
import matplotlib.pyplot as plt

n_eff = 2.45
radius = 10e-6
kappa = 0.45
a = 0.98

L = 2 * np.pi * radius
r = np.sqrt(1 - kappa**2)
wavelength = np.linspace(1.4e-6, 1.6e-6, 10000)

phi = (2 * np.pi * n_eff * L) / wavelength
T = (a**2 + r**2 - 2 * a * r * np.cos(phi)) / (1 + (a * r)**2 - 2 * a * r * np.cos(phi))

plt.figure(figsize=(10, 5))
plt.plot(wavelength * 1e9, T, linewidth=1.5, color='#0072BD')
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Transmission (a.u.)', fontsize=12)
plt.title('MRR Transmission Spectrum', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()