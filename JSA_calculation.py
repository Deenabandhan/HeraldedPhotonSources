import numpy as np
import matplotlib.pyplot as plt

def alpha(w, dw):
    return np.exp(-0.5 * (w / dw)**2)

def l_cav(w, dw):
    return dw / (dw - 1j * w)

def compute_jsa_and_purity(dw_p_cav, dw_s_cav, dw_i_cav):
    N = 120
    w = np.linspace(-6, 6, N)
    ws, wi = np.meshgrid(w, w)
    
    wp = np.linspace(-15, 15, 250)
    dwp = wp[1] - wp[0]
    
    dw_p_env = 6.0
    
    W_sum = (ws + wi)[..., np.newaxis]
    WP = wp[np.newaxis, np.newaxis, :]
    
    integrand = alpha(WP, dw_p_env) * alpha(W_sum - WP, dw_p_env) * \
                l_cav(WP, dw_p_cav) * l_cav(W_sum - WP, dw_p_cav)
    
    integral_res = np.sum(integrand, axis=2) * dwp
    
    F = integral_res * l_cav(ws, dw_s_cav) * l_cav(wi, dw_i_cav)
    
    U, S, Vh = np.linalg.svd(F)
    S_sq = S**2
    lambda_n = S_sq / np.sum(S_sq)
    purity = np.sum(lambda_n**2)
    
    return ws, wi, np.abs(F)**2, purity

ws, wi, F2_entangled, pur_entangled = compute_jsa_and_purity(dw_p_cav=0.5, dw_s_cav=4.0, dw_i_cav=4.0)

ws, wi, F2_pure, pur_pure = compute_jsa_and_purity(dw_p_cav=6.0, dw_s_cav=0.5, dw_i_cav=0.5)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

c1 = axes[0].contourf(ws, wi, F2_entangled, 50, cmap='viridis')
axes[0].set_title(rf'Entangled State ($Q_p \gg Q_s, Q_i$)' + '\n' + rf'Schmidt Purity = {pur_entangled*100:.2f}%', fontsize=13)
axes[0].set_xlabel(r'Signal Detuning ($\omega_s$)', fontsize=12)
axes[0].set_ylabel(r'Idler Detuning ($\omega_i$)', fontsize=12)
axes[0].set_aspect('equal')
fig.colorbar(c1, ax=axes[0], fraction=0.046, pad=0.04)

c2 = axes[1].contourf(ws, wi, F2_pure, 50, cmap='viridis')
axes[1].set_title(rf'Pure State ($Q_p \ll Q_s, Q_i$)' + '\n' + rf'Schmidt Purity = {pur_pure*100:.2f}%', fontsize=13)
axes[1].set_xlabel(r'Signal Detuning ($\omega_s$)', fontsize=12)
axes[1].set_ylabel(r'Idler Detuning ($\omega_i$)', fontsize=12)
axes[1].set_aspect('equal')
fig.colorbar(c2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()