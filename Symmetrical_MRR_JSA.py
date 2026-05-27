import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# SYMMETRIC CHIP ANALYSIS (data_pls.dat)
# ==========================================
filename = 'data_pls.dat'
offset = 40e-9
a = 0.99 # 1% internal loss
n_eff = 2.45
L_ring = 2 * np.pi * 10e-6

# 1. Data Loading & Cleaning
data = np.loadtxt(filename, delimiter=',')
wavelength = data[:, 0]
raw_k_sq = data[:, 1]

sort_idx = np.argsort(wavelength)
wavelength = wavelength[sort_idx]
raw_k_sq = raw_k_sq[sort_idx]

win_len = max(3, (len(wavelength) // 10) | 1)
smooth_k_sq = savgol_filter(raw_k_sq, win_len, 3)
noise_floor = np.min(smooth_k_sq)
smooth_k_sq = np.clip(smooth_k_sq - noise_floor, 0.0001, 0.999)

# 2. Extract Physics at Maximum Peak
idx_p = np.argmax(smooth_k_sq)
lam_p = wavelength[idx_p]
k2_p = smooth_k_sq[idx_p]

idx_s = np.argmin(np.abs(wavelength - (lam_p + offset)))
idx_i = np.argmin(np.abs(wavelength - (lam_p - offset)))
k2_s = smooth_k_sq[idx_s]
k2_i = smooth_k_sq[idx_i]

r_p, r_s, r_i = np.sqrt(1 - k2_p), np.sqrt(1 - k2_s), np.sqrt(1 - k2_i)

print(f"--- SYMMETRIC RESULTS ---")
print(f"Pump Peak Found At: {lam_p*1e9:.1f} nm")
print(f"Pump k^2: {k2_p:.4f} | Signal k^2: {k2_s:.4f}")

# 3. Transmission Spectrum Plot
r_array = np.sqrt(1 - smooth_k_sq)
phi = (2 * np.pi * n_eff * L_ring) / wavelength
T = (a**2 + r_array**2 - 2 * a * r_array * np.cos(phi)) / (1 + (a * r_array)**2 - 2 * a * r_array * np.cos(phi))

plt.figure(figsize=(8, 4))
plt.plot(wavelength * 1e9, T, color='#0072BD', linewidth=1.5)
plt.title('Symmetric Design: Transmission Spectrum', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Transmission (T)', fontsize=12)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# 4. JSA & Purity Math
base_scale = 0.5 / (1 - a * 0.9999) 
dw_p = (1 - a * r_p) * base_scale
dw_s = (1 - a * r_s) * base_scale
dw_i = (1 - a * r_i) * base_scale

def alpha(w, dw): return np.exp(-0.5 * (w / dw)**2)
def l_cav(w, dw): return dw / (dw - 1j * w)

w = np.linspace(-8, 8, 120)
ws, wi = np.meshgrid(w, w)
wp = np.linspace(-20, 20, 200)
dwp = wp[1] - wp[0]

WP = wp[np.newaxis, np.newaxis, :]
W_sum = (ws + wi)[..., np.newaxis]
dw_p_env = 6.0

integrand = alpha(WP, dw_p_env) * alpha(W_sum - WP, dw_p_env) * l_cav(WP, dw_p) * l_cav(W_sum - WP, dw_p)
F = np.sum(integrand, axis=2) * dwp * l_cav(ws, dw_s) * l_cav(wi, dw_i)

U, S, Vh = np.linalg.svd(F)
purity = np.sum((S**2 / np.sum(S**2))**2)
print(f"Schmidt Purity: {purity*100:.1f}%\n")

# 5. JSA Plot
plt.figure(figsize=(6, 5))
c = plt.contourf(ws, wi, np.abs(F)**2, 50, cmap='inferno')
plt.title(f'Symmetric JSA\nPurity: {purity*100:.1f}%', fontsize=14)
plt.xlabel(r'Signal Detuning ($\omega_s$)', fontsize=12)
plt.ylabel(r'Idler Detuning ($\omega_i$)', fontsize=12)
plt.colorbar(c, label='Intensity')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()