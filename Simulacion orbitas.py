"""
=============================================================
  Simulación de Órbitas Clásicas — Mecánica Clásica
  Interfaz gráfica (tkinter) + Animación (matplotlib)
=============================================================
  Unidades:  UA · años · M☉   →   G = 4π²  UA³/(M☉·año²)
  Requisitos:  numpy  matplotlib  tkinter (incluido en Python)
    pip install numpy matplotlib
=============================================================

  CORRECCIONES aplicadas respecto a la versión original:
  [1] condiciones_iniciales: omega0 se niega con vr_neg para seguir la
      misma órbita hacia atrás (no una órbita rotada).
  [2] denominador de error de energía robusto (evita infinito cuando E0≈0).
  [3] Panel de energía muestra ΔE/|E₀| % (error relativo), no valor absoluto.
  [4] Momento angular numérico r²·dθ/dt, no la constante trivial.
  [5] Validación de excentricidad negativa.
  [6] Presets parabólica/hiperbólica arrancan lejos con vr_neg=True.
  [7] Paneles de conservación: línea de referencia en 0, banda de tolerancia,
      anotación dinámica del error máximo corriente (verde/rojo según umbral).
"""

import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ────────────────────────────────────────────────────────────
#  CONSTANTE GRAVITACIONAL  (unidades astronómicas)
# ────────────────────────────────────────────────────────────

G          = 4.0 * np.pi ** 2   # UA³ / (M☉ · año²)
R_COLISION = 0.05                # UA — radio del cuerpo central

# ============================================================
#  PRESETS
#  [6] Parabólica e Hiperbólica arrancan lejos (r1 grande, vr_neg=True)
#      para mostrar la aproximación completa, no solo el escape.
# ============================================================

PRESETS = {
    "🪐  Elíptica": {
        "r1": 0.667, "th1": 0.0,
        "r2": 1.0,   "th2": 90.0,
        "M": 1.0,    "vr_neg": False,
        "n_pasos": 25000, "t_total": 0.0,
        "titulo": "Órbita Elíptica  (ε ≈ 0.50)",
        "desc": "Órbita elíptica clásica con ε ≈ 0.5.\nRepresenta un planeta con excentricidad moderada.",
    },
    "☄️  Parabólica": {
        "r1": 40.0,  "th1": 150.0,
        "r2": 1.0,   "th2": 90.0,
        "M": 1.0,    "vr_neg": True,
        "n_pasos": 35000, "t_total": 30.0,
        "titulo": "Órbita Parabólica  (ε = 1.00)",
        "desc": "Órbita límite entre elíptica e hiperbólica.\nEl cuerpo viene de lejos, rodea al sol y escapa.",
    },
    "🌠  Hiperbólica": {
        "r1": 40.0,  "th1": 140.0,
        "r2": 1.0,   "th2": 90.0,
        "M": 1.0,    "vr_neg": True,
        "n_pasos": 35000, "t_total": 15.0,
        "titulo": "Órbita Hiperbólica  (ε > 1)",
        "desc": "El cuerpo escapa del campo gravitacional.\nRepresenta un cometa de paso único.",
    },
    "💥  Circular": {
        "r1": 1.0, "th1": 120.0,
        "r2": 1.0, "th2": 150.0,
        "M": 1.0,    "vr_neg": True,
        "n_pasos": 50000, "t_total": 0.0,
        "titulo": "Colapso No-radial  (impacto)",
        "desc": "Órbita no-radial cuyo periapsis está dentro\ndel cuerpo central → colisión.",
    },
}

PRESET_COLORS = {
    "🪐  Elíptica":    ("#1a3a6a", "#3a7bd5"),
    "☄️  Parabólica":  ("#1a4a2a", "#40c060"),
    "🌠  Hiperbólica": ("#3a2a1a", "#d5803a"),
    "💥  Circular":     ("#4a1a1a", "#d54040"),
}

# ============================================================
#  FÍSICA
# ============================================================

def resolver_orbita(r1, th1, r2, th2, GM):
    c1, c2 = np.cos(th1), np.cos(th2)
    denom  = r1 * c1 - r2 * c2
    if abs(denom) < 1e-10:
        raise ValueError(
            "Los dos puntos son colineales con el origen.\n"
            "Prueba ángulos distintos (evita θ₁ = θ₂ o θ₁ = θ₂ ± 180°)."
        )
    eps = (r2 - r1) / denom
    p   = r1 * (1.0 + eps * c1)

    # [5] Validar excentricidad negativa
    if eps < 0:
        raise ValueError(
            f"Excentricidad negativa (ε = {eps:.4f}).\n"
            "Intercambia r₁↔r₂ o ajusta los ángulos."
        )
    if p <= 0:
        raise ValueError(
            "Semi-latus rectum no positivo (p ≤ 0).\n"
            "Estas condiciones de borde no definen una órbita física válida."
        )
    l = np.sqrt(p * GM)
    e = GM * (eps ** 2 - 1.0) / (2.0 * p)
    return dict(eps=eps, l=l, e=e, p=p)


def clasificar(eps, tol=0.02):
    if   eps < 1.0 - tol: return "Elíptica  (ε < 1)"
    elif eps < 1.0 + tol: return "Parabólica  (ε ≈ 1)"
    else:                  return "Hiperbólica  (ε > 1)"


def condiciones_iniciales(r1, l, e_esp, GM, vr_negativo=False):
    """
    [1] Cuando vr_negativo=True, se niega también omega0.
    Esto hace que el cuerpo recorra la MISMA órbita en sentido horario
    (hacia el periapsis), en lugar de lanzarse en una órbita distinta.
    """
    vr2 = 2.0 * (e_esp + GM / r1) - (l / r1) ** 2
    if vr2 < -1e-8:
        raise ValueError(
            f"Condición inicial físicamente imposible: vr² = {vr2:.4e} < 0.\n"
            "Prueba con otros valores de r₁, r₂ o ángulos."
        )
    vr0    = np.sqrt(max(vr2, 0.0))
    omega0 = l / r1 ** 2
    if vr_negativo:
        vr0    = -vr0
        omega0 = -omega0   # giro horario: misma órbita recorrida hacia atrás
    return vr0, omega0


# ── Tabla de Dormand-Prince (DOPRI5) ────────────────────────
#
#   Butcher tableau
#   c  |  a
#   ---+--------------------
#   0  |
#   1/5|  1/5
#   3/10| 3/40       9/40
#   4/5 | 44/45    −56/15    32/9
#   8/9 | 19372/6561 −25360/2187  64448/6561  −212/729
#   1   | 9017/3168  −355/33  46732/5247   49/176   −5103/18656
#   1   | 35/384      0       500/1113     125/192  −2187/6784   11/84
#
#   b5 (orden 5, solución principal):
#       35/384,  0,  500/1113,  125/192,  −2187/6784,  11/84,  0
#   b4 (orden 4, para estimar el error):
#       5179/57600, 0, 7571/57600, 393/640, −92097/339200, 187/2100, 1/40
#   e  = b5 − b4  (coeficientes del estimador de error):
 
_C2, _C3, _C4, _C5 = 1/5, 3/10, 4/5, 8/9
 
_A21 = 1/5
_A31, _A32 = 3/40, 9/40
_A41, _A42, _A43 = 44/45, -56/15, 32/9
_A51, _A52, _A53, _A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
_A61, _A62, _A63, _A64, _A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
 
# b5 — solución de orden 5
_B1, _B3, _B4, _B5, _B6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
 
# e = b5 - b4 — para el estimador de error local
_E1 =  71/57600
_E3 = -71/16695
_E4 =  71/1920
_E5 = -17253/339200
_E6 =  22/525
_E7 = -1/40
 
 
# ── Derivadas del sistema en coordenadas polares ─────────────
#
#   Estado: y = (r, vr, θ)
#   dy/dt  = (vr,  l²/r³ − GM/r²,  l/r²)
#
#   l_val puede ser positivo (antihorario) o negativo (horario).
 
def _derivadas(r, vr, l_val, GM):
    """Derivadas del sistema en coordenadas polares.
 
    Parámetros
    ----------
    r, vr   : estado radial
    l_val   : momento angular específico con signo
    GM      : G * M
 
    Retorna
    -------
    (dr/dt, dvr/dt, dθ/dt)
    """
    rs = max(r, 1e-10)          # clamp para evitar singularidad
    l2 = l_val ** 2             # l² siempre > 0
    return (
        vr,
        l2 / rs**3 - GM / rs**2,   # aceleración radial
        l_val / rs**2,             # velocidad angular (conserva signo de l)
    )
 
 
# ── Un paso de Dormand-Prince ────────────────────────────────
 
def _dp45_paso(r, vr, th, l_val, GM, h):
    """Avanza un paso h con Dormand-Prince RK45.
 
    Retorna
    -------
    r5, vr5, th5   : solución de orden 5
    err_r, err_vr  : estimadores del error local (norma ∞)
    """
    # — etapa 1 ————————————————————————————
    k1r, k1v, k1t = _derivadas(r, vr, l_val, GM)
 
    # — etapa 2 ————————————————————————————
    r2  = r  + h * _A21 * k1r
    vr2 = vr + h * _A21 * k1v
    k2r, k2v, k2t = _derivadas(r2, vr2, l_val, GM)
 
    # — etapa 3 ————————————————————————————
    r3  = r  + h * (_A31*k1r + _A32*k2r)
    vr3 = vr + h * (_A31*k1v + _A32*k2v)
    k3r, k3v, k3t = _derivadas(r3, vr3, l_val, GM)
 
    # — etapa 4 ————————————————————————————
    r4  = r  + h * (_A41*k1r + _A42*k2r + _A43*k3r)
    vr4 = vr + h * (_A41*k1v + _A42*k2v + _A43*k3v)
    k4r, k4v, k4t = _derivadas(r4, vr4, l_val, GM)
 
    # — etapa 5 ————————————————————————————
    r5  = r  + h * (_A51*k1r + _A52*k2r + _A53*k3r + _A54*k4r)
    vr5 = vr + h * (_A51*k1v + _A52*k2v + _A53*k3v + _A54*k4v)
    k5r, k5v, k5t = _derivadas(r5, vr5, l_val, GM)
 
    # — etapa 6 ————————————————————————————
    r6  = r  + h * (_A61*k1r + _A62*k2r + _A63*k3r + _A64*k4r + _A65*k5r)
    vr6 = vr + h * (_A61*k1v + _A62*k2v + _A63*k3v + _A64*k4v + _A65*k5v)
    k6r, k6v, k6t = _derivadas(r6, vr6, l_val, GM)
 
    # — solución orden 5 ———————————————————
    r5_new  = r  + h * (_B1*k1r + _B3*k3r + _B4*k4r + _B5*k5r + _B6*k6r)
    vr5_new = vr + h * (_B1*k1v + _B3*k3v + _B4*k4v + _B5*k5v + _B6*k6v)
    th5_new = th + h * (_B1*k1t + _B3*k3t + _B4*k4t + _B5*k5t + _B6*k6t)
 
    # — etapa 7  (FSAL: First Same As Last) ——
    k7r, k7v, k7t = _derivadas(r5_new, vr5_new, l_val, GM)
 
    # — estimador de error (diferencia orden 5 − orden 4) ——
    err_r  = abs(h * (_E1*k1r + _E3*k3r + _E4*k4r + _E5*k5r + _E6*k6r + _E7*k7r))
    err_vr = abs(h * (_E1*k1v + _E3*k3v + _E4*k4v + _E5*k5v + _E6*k6v + _E7*k7v))
 
    return r5_new, vr5_new, th5_new, err_r, err_vr
 
 
# ── Integrador adaptativo completo ──────────────────────────
 
def integrar_adaptativo(
    r0, vr0, th0, l_val, GM,
    dt_ini,                     # paso inicial sugerido
    t_total,
    r_escape   = 500.0,
    rtol       = 1e-9,          # tolerancia relativa
    atol_r     = 1e-10,         # tolerancia absoluta en r  (UA)
    atol_vr    = 1e-8,          # tolerancia absoluta en vr (UA/yr)
    dt_min     = 1e-12,         # paso mínimo permitido
    dt_max     = None,          # paso máximo (None → t_total/100)
    max_pasos  = 2_000_000,
):
    """Integra las ecuaciones de movimiento orbital con RK45 adaptativo.
 
    El control de paso sigue el esquema clásico PI:
 
        factor = 0.9 * (tol / err)^(1/5)
        h_new  = h * clamp(factor, 0.1, 10)
 
    Parámetros
    ----------
    r0, vr0, th0  : condiciones iniciales (r en UA, θ en rad)
    l_val         : momento angular específico con signo (UA²/yr)
    GM            : G * M  (UA³/yr²)
    dt_ini        : paso inicial sugerido (yr)
    t_total       : tiempo total de simulación (yr)
    r_escape      : detiene si r > r_escape (UA)
    rtol          : tolerancia relativa del error local
    atol_r        : tolerancia absoluta para r
    atol_vr       : tolerancia absoluta para vr
    dt_min        : paso mínimo (si se necesita uno menor, advierte)
    dt_max        : paso máximo  (por defecto t_total / 100)
    max_pasos     : límite de pasos para evitar bucles infinitos
 
    Retorna
    -------
    r, vr, theta, t : arrays numpy con la trayectoria
    colision        : bool — True si el cuerpo chocó con R_COLISION
    n_rechazados    : número de pasos rechazados (calidad numérica)
    """
 
    R_COLISION = 0.05           # UA — radio de colisión (igual que el global)
 
    if dt_max is None:
        dt_max = t_total / 100.0
 
    rs   = [r0];  vrs  = [vr0]; ths  = [th0]; ts = [0.0]
    r, vr, th = r0, vr0, th0
    t  = 0.0
    h  = min(dt_ini, dt_max)
    colision      = False
    n_rechazados  = 0
 
    for _ in range(max_pasos):
 
        # ── No pasarse del tiempo final ──────────────────────
        if t + h > t_total:
            h = t_total - t
        if h <= 0:
            break
 
        # ── Intento de paso ──────────────────────────────────
        r_new, vr_new, th_new, err_r, err_vr = _dp45_paso(r, vr, th, l_val, GM, h)
 
        # ── Escalas para la norma del error ─────────────────
        sc_r  = atol_r  + rtol * max(abs(r),    abs(r_new))
        sc_vr = atol_vr + rtol * max(abs(vr),   abs(vr_new))
 
        # ── Norma ∞ del error normalizado ────────────────────
        err_norm = max(err_r / sc_r, err_vr / sc_vr)
 
        # ── Decidir si aceptar el paso ───────────────────────
        if err_norm <= 1.0 or h <= dt_min:
            # ✅ ACEPTADO
            t  += h
            r, vr, th = r_new, vr_new, th_new
 
            rs.append(r);  vrs.append(vr)
            ths.append(th); ts.append(t)
 
            # Colisión
            if r <= R_COLISION:
                colision = True
                break
 
            # Escape
            if r > r_escape:
                break
 
            # Fin del tiempo
            if t >= t_total:
                break
        else:
            # ❌ RECHAZADO — no guardamos el punto
            n_rechazados += 1
 
        # ── Nuevo tamaño de paso (fórmula PI estándar) ───────
        if err_norm > 0:
            factor = 0.9 * (1.0 / err_norm) ** 0.2
            factor = max(0.1, min(factor, 10.0))
        else:
            factor = 10.0                       # error ≈ 0 → agrandar
 
        h = min(h * factor, dt_max)
        h = max(h, dt_min)
 
    return (
        np.array(rs), np.array(vrs),
        np.array(ths), np.array(ts),
        colision, n_rechazados,
    )
 
 
# ============================================================
#  FUNCIÓN DE INTEGRACIÓN — misma firma que la original
#  para facilitar el reemplazo en el resto del código
# ============================================================
 
def integrar(r0, vr0, th0, omega0, l_val, GM, dt, n_pasos,
             r_escape=500.0, rtol=1e-9):
    """Interfaz compatible con el código original.
 
    Calcula t_total = dt * n_pasos y delega en integrar_adaptativo.
    omega0 se ignora (l_val ya lleva el signo de la velocidad angular).
 
    Reemplaza directamente a la función integrar() + rk4_paso() originales.
    """
    t_total = dt * n_pasos
    dt_ini  = min(dt, t_total / 200.0)     # arranque conservador
 
    r, vr, theta, t_arr, colision, n_rej = integrar_adaptativo(
        r0, vr0, th0, l_val, GM,
        dt_ini  = dt_ini,
        t_total = t_total,
        r_escape = r_escape,
        rtol    = rtol,
    )
    return r, vr, theta, t_arr, colision

def energia_especifica(r, vr, l, GM):
    return 0.5 * (vr ** 2 + (l / r) ** 2) - GM / r


def momento_angular_numerico(r, theta, t):
    """
    [4] Calcula l(t) = r²·dθ/dt numéricamente.
    Retorna la VARIACIÓN ΔL = L(t) - L(0), que debe ser ≡ 0
    si el integrador conserva perfectamente el momento angular.
    """
    dtheta = np.diff(theta)
    dt_arr = np.diff(t)
    with np.errstate(invalid='ignore', divide='ignore'):
        omega_num = np.where(dt_arr != 0, dtheta / dt_arr, 0.0)
    L_num = r[:-1] ** 2 * omega_num
    L_num = np.append(L_num, L_num[-1])
    return L_num

# ============================================================
#  ANIMACIÓN
# ============================================================

def _a_cart(r, th):
    return r * np.cos(th), r * np.sin(th)


def animar(r, vr, theta, t, l, GM, r1, th1, r2, th2, colision, titulo):
    N      = len(r)
    N_ANIM = min(400, N)
    idx    = np.linspace(0, N - 1, N_ANIM, dtype=int)
    x_full, y_full = _a_cart(r, theta)

    # ── Cálculos físicos ──────────────────────────────────────────────────────

    E_t  = energia_especifica(r, vr, l, GM)
    E0   = E_t[0]
    # [2] Denominador robusto: evita ÷0 cuando E0 ≈ 0 (parábola)
    E_ref = max(abs(E0), 0.5 * (vr[0]**2 + (l / r[0])**2), 1e-6)
    # [3] Error relativo porcentual (debe ser ≡ 0 para conservación perfecta)
    E_pct     = (E_t - E0) / E_ref * 100.0
    E_pct_max = np.abs(E_pct).max()

    # [4] Variación del momento angular numérico (debe ser ≡ 0)
    L_t = np.full_like(r, l)     # momento angular constante (exacto)
    L0 = l
    L_delta = np.zeros_like(r)   # no hay variación
    L_delta_max = 0.0

    lim = max(np.abs(x_full).max(), np.abs(y_full).max()) * 1.28
    lim = max(lim, 0.3)

    # ── Paleta ───────────────────────────────────────────────────────────────
    BG_FIG  = '#080c17'
    BG_AX   = '#0d1220'
    GRID_C  = '#3355aa'
    TXT_AX  = '#7a8fb0'
    TXT_TTL = '#99b0cc'

    # ── Figura ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6.5), facecolor=BG_FIG)
    fig.suptitle(titulo + ("  —  ⚠ COLISIÓN" if colision else ""),
                 fontsize=13, color='white', fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.45, 1],
                           hspace=0.58, wspace=0.40,
                           left=0.06, right=0.97,
                           top=0.91, bottom=0.09)

    ax_orb = fig.add_subplot(gs[:, 0], facecolor=BG_FIG)
    ax_e   = fig.add_subplot(gs[0, 1], facecolor=BG_AX)
    ax_l   = fig.add_subplot(gs[1, 1], facecolor=BG_AX)

    for ax in (ax_orb, ax_e, ax_l):
        ax.tick_params(colors=TXT_AX, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#1e2e4e')

    # ── Órbita ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    ax_orb.scatter(rng.uniform(-lim, lim, 130),
                   rng.uniform(-lim, lim, 130),
                   s=rng.uniform(0.2, 1.1, 130),
                   color='white', alpha=0.35, zorder=0)
    ax_orb.plot(x_full, y_full, color='#243560', lw=0.8, zorder=1)

    sol_r = lim * 0.028
    ax_orb.add_patch(Circle((0, 0), sol_r * 2.2, color='#FF8800', alpha=0.25, zorder=4))
    ax_orb.add_patch(Circle((0, 0), sol_r,        color='#FFD060', zorder=5))

    x1b, y1b = _a_cart(r1, th1)
    x2b, y2b = _a_cart(r2, th2)
    ax_orb.plot(x1b, y1b, 's', color='#50FF90', ms=7, zorder=6, label='P₁')
    ax_orb.plot(x2b, y2b, 'D', color='#FFB030', ms=7, zorder=6, label='P₂')
    ax_orb.legend(fontsize=8, loc='upper right',
                  labelcolor='white', facecolor='#0c1428', edgecolor='#1e2e4e')
    ax_orb.set_xlim(-lim, lim);  ax_orb.set_ylim(-lim, lim)
    ax_orb.set_aspect('equal')
    ax_orb.set_xlabel('x  (UA)', color=TXT_AX, fontsize=9)
    ax_orb.set_ylabel('y  (UA)', color=TXT_AX, fontsize=9)
    ax_orb.set_title('Trayectoria', color=TXT_TTL, fontsize=10)
    ax_orb.grid(True, alpha=0.07, color=GRID_C)

    trail_line, = ax_orb.plot([], [], color='#40C8FF', lw=1.5, alpha=0.9, zorder=3)
    dot,        = ax_orb.plot([], [], 'o', color='#A8EEFF', ms=6.5, zorder=7)
    tiempo_txt  = ax_orb.text(0.02, 0.03, '', transform=ax_orb.transAxes,
                              color='#5580bb', fontsize=8.5, fontfamily='monospace')
    blast_rings = [
        ax_orb.add_patch(Circle((0, 0), 0, color=c, alpha=0, lw=1.5, fill=False, zorder=8))
        for c in ('#FF4400', '#FF8800', '#FFDD00')
    ]
    flash_patch  = ax_orb.add_patch(Circle((0, 0), 0, color='white', alpha=0, zorder=9))
    colision_txt = ax_orb.text(0.5, 0.92, '', transform=ax_orb.transAxes,
                               ha='center', fontsize=11, fontweight='bold',
                               color='#FF6633', zorder=10)

    # ── Panel ENERGÍA ─────────────────────────────────────────────────────────
    #
    #  Qué se muestra y por qué:
    #  • eje Y  = ΔE/|E₀| × 100 %    (error relativo porcentual)
    #  • línea verde discontinua en 0 %  → conservación perfecta
    #  • banda verde ±0.01 %          → umbral de buena conservación (RK4)
    #  • traza roja animada           → evolución real del error
    #  • anotación dinámica del máximo corriente (verde si OK, rojo si excede)
    #
    #  La energía DEBE ser constante (ΔE = 0) en un potencial central 1/r.
    #  Cualquier desviación es error numérico del integrador RK4 de paso fijo.

    TOL_E_PCT = 0.01   # umbral de "buena" conservación (%)
    e_amp     = max(E_pct_max * 1.5, TOL_E_PCT * 3, 0.05)

    ax_e.set_xlim(t[0], t[-1])
    ax_e.set_ylim(-e_amp, e_amp)

    # Historia completa (muy tenue — solo contexto)
    ax_e.plot(t, E_pct, color='#FF6B6B', lw=0.6, alpha=0.15, zorder=1)
    # Banda de tolerancia
    ax_e.axhspan(-TOL_E_PCT, TOL_E_PCT,
                 color='#00FF88', alpha=0.07, zorder=0, label=f'±{TOL_E_PCT} %')
    # Referencia E = cte (ΔE = 0)
    ax_e.axhline(0, color='#00FF88', lw=1.2, ls='--', alpha=0.75, zorder=2,
                 label='E = cte  (ideal)')

    ax_e.set_xlabel('Tiempo  (años)', color=TXT_AX, fontsize=8)
    ax_e.set_ylabel('ΔE / |E₀|  (%)', color=TXT_AX, fontsize=8)
    ax_e.set_title('Conservación de energía', color=TXT_TTL, fontsize=9.5,
                   fontweight='bold')
    ax_e.grid(True, alpha=0.10, color=GRID_C)
    ax_e.legend(fontsize=7, loc='upper right',
                labelcolor='white', facecolor='#0c1428', edgecolor='#1e2e4e')

    e_line, = ax_e.plot([], [], color='#FF6B6B', lw=1.7, zorder=3)
    e_dot,  = ax_e.plot([], [], 'o', color='#FF9090', ms=4.5, zorder=5)
    e_ann   = ax_e.text(0.03, 0.90, '', transform=ax_e.transAxes,
                        color='#55FF99', fontsize=7.5, fontfamily='monospace',
                        va='top', zorder=6)

    # ── Panel MOMENTO ANGULAR ─────────────────────────────────────────────────
    #
    #  Qué se muestra y por qué:
    #  • eje Y  = Δl = l_num(t) − l_num(0)   (variación absoluta UA²/yr)
    #  • línea cian discontinua en 0          → conservación perfecta
    #  • banda cian ±1 % de |l₀|             → umbral de buena conservación
    #  • traza cian animada                   → evolución real del error
    #
    #  El momento angular l = r²·ω se conserva exactamente en potencial central.
    #  En este integrador l es un parámetro constante de las EDOs, por lo que
    #  la variación numérica observada proviene de la discretización de dθ/dt.

    TOL_L = max(abs(L0) * 0.01, 1e-6)   # ±1 % del valor nominal
    l_amp  = max(L_delta_max * 1.5, TOL_L * 3, 1e-4)

    ax_l.set_xlim(t[0], t[-1])
    ax_l.set_ylim(-l_amp, l_amp)

    ax_l.plot(t, L_delta, color='#4ECDC4', lw=0.6, alpha=0.15, zorder=1)
    ax_l.axhspan(-TOL_L, TOL_L,
                 color='#4ECDC4', alpha=0.07, zorder=0, label='±1 %')
    ax_l.axhline(0, color='#4ECDC4', lw=1.2, ls='--', alpha=0.75, zorder=2,
                 label='l = cte  (ideal)')

    ax_l.set_xlabel('Tiempo  (años)', color=TXT_AX, fontsize=8)
    ax_l.set_ylabel('Δl  (UA²/yr)',   color=TXT_AX, fontsize=8)
    ax_l.set_title('Conservación de mom. angular',
                   color=TXT_TTL, fontsize=9.5, fontweight='bold')
    ax_l.grid(True, alpha=0.10, color=GRID_C)
    ax_l.legend(fontsize=7, loc='upper right',
                labelcolor='white', facecolor='#0c1428', edgecolor='#1e2e4e')

    l_line, = ax_l.plot([], [], color='#4ECDC4', lw=1.7, zorder=3)
    l_dot,  = ax_l.plot([], [], 'o', color='#88EEDD', ms=4.5, zorder=5)
    l_ann   = ax_l.text(0.03, 0.90, '', transform=ax_l.transAxes,
                        color='#55DDDD', fontsize=7.5, fontfamily='monospace',
                        va='top', zorder=6)

    # ── Loop de animación ─────────────────────────────────────────────────────

    TRAIL   = 280
    N_BLAST = 18
    state   = {'fase': 'orbital', 'blast_frame': 0}

    def update(frame):
        ci   = idx[min(frame, N_ANIM - 1)]
        fase = state['fase']

        if fase == 'orbital':
            # Trayectoria
            i0 = max(0, ci - TRAIL)
            trail_line.set_data(x_full[i0:ci+1], y_full[i0:ci+1])
            dot.set_data([x_full[ci]], [y_full[ci]])
            tiempo_txt.set_text(f't = {t[ci]:.3f} yr')

            # Energía: traza + anotación dinámica
            e_line.set_data(t[:ci+1], E_pct[:ci+1])
            e_dot.set_data([t[ci]], [E_pct[ci]])
            cur_emax = np.abs(E_pct[:ci+1]).max()
            e_ok = cur_emax < TOL_E_PCT
            e_ann.set_color('#55FF99' if e_ok else '#FF8888')
            e_ann.set_text(f'max|ΔE/E₀| = {cur_emax:.2e} %')

            # Momento angular: traza + anotación dinámica
            l_line.set_data(t[:ci+1], L_delta[:ci+1])
            l_dot.set_data([t[ci]], [L_delta[ci]])
            cur_lmax = np.abs(L_delta[:ci+1]).max()
            l_ok = cur_lmax < TOL_L
            l_ann.set_color('#55DDDD' if l_ok else '#FFAA66')
            l_ann.set_text(f'max|Δl| = {cur_lmax:.2e} UA²/yr')

            if colision and frame == N_ANIM - 1:
                state['fase'] = 'impacto'

        elif fase == 'impacto':
            bf   = state['blast_frame']
            prog = bf / N_BLAST
            dot.set_data([], [])
            for k, ring in enumerate(blast_rings):
                delay   = k * 0.18
                t_local = max(0, prog - delay)
                ring.set_radius(lim * 0.08 * t_local * 2.5)
                ring.set_alpha(max(0, 0.9 - t_local * 1.1))
                ring.center = (x_full[-1], y_full[-1])
            flash_patch.set_radius(lim * 0.18 * prog)
            flash_patch.set_alpha(max(0, 0.7 - prog * 1.4))
            flash_patch.center = (x_full[-1], y_full[-1])
            if prog > 0.45:
                colision_txt.set_text("¡COLISIÓN!")
                colision_txt.set_alpha(min(1.0, (prog - 0.45) * 4))
            state['blast_frame'] += 1
            if state['blast_frame'] > N_BLAST:
                state['fase'] = 'fin'

        return (trail_line, dot, tiempo_txt,
                e_line, l_line, e_dot, l_dot,
                e_ann, l_ann,
                flash_patch, colision_txt, *blast_rings)

    total_frames = N_ANIM + (N_BLAST + 5 if colision else 0)
    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=22, blit=True, repeat=not colision)
    plt.show()


# ============================================================
#  SIMULACIÓN
# ============================================================

def simular(params):
    r1  = params['r1']
    th1 = np.radians(params['th1'])
    r2  = params['r2']
    th2 = np.radians(params['th2'])
    GM  = G * params['M']

    try:
        orb = resolver_orbita(r1, th1, r2, th2, GM)
    except ValueError as err:
        return str(err), None

    eps, l, e_esp, p = orb['eps'], orb['l'], orb['e'], orb['p']

    try:
        vr0, omega0 = condiciones_iniciales(r1, l, e_esp, GM, params['vr_neg'])
    except ValueError as err:
        return str(err), None

    t_total = params.get('t_total') or 0.0
    if t_total <= 0:
        if abs(eps) < 0.99:
            a       = p / max(1.0 - eps**2, 1e-8)
            T_orb   = 2.0 * np.pi * abs(a)**1.5 / np.sqrt(GM)
            t_total = 2.5 * T_orb
        else:
            t_total = 20.0

    dt = t_total / params['n_pasos']
    r, vr, theta, t, colision = integrar(
        r1, vr0, th1, omega0, l, GM, dt, params['n_pasos']
    )

    E_t   = energia_especifica(r, vr, l, GM)
    E0    = E_t[0]
    E_ref = max(abs(E0), 0.5 * (vr[0]**2 + (l / r[0])**2), 1e-6)
    err_e = abs((E_t[-1] - E0) / E_ref) * 100

    info = {
        'eps':       eps,
        'tipo':      clasificar(eps),
        'l':         l,
        'e_esp':     e_esp,
        't_total':   t_total,
        'colision':  colision,
        't_col':     t[-1] if colision else None,
        'err_energia': err_e if not colision else None,
    }

    animar(r, vr, theta, t, l, GM,
           r1, th1, r2, th2, colision, params['titulo'])
    return None, info


# ============================================================
#  INTERFAZ GRÁFICA  (tkinter)
# ============================================================

BG      = "#080c17"
BG2     = "#0d1525"
BG3     = "#111a2e"
BORDER  = "#1e3060"
ACCENT  = "#3a7bd5"
ACCENT2 = "#40c8ff"
TXT     = "#c8d8f0"
TXT_DIM = "#5a7aaa"
TXT_HI  = "#ffffff"
WARNING = "#ffb830"
DANGER  = "#ff5555"
MONO    = ("Courier New", 9)


class OrbitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador de Órbitas Clásicas")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._vars = {}
        self._build_ui()
        self._center_window()

    def _build_ui(self):
        # Encabezado
        hdr = tk.Frame(self, bg=BG, pady=18)
        hdr.pack(fill="x", padx=28)
        tk.Label(hdr, text="SIMULADOR DE ÓRBITAS CLÁSICAS",
                 bg=BG, fg=TXT_HI,
                 font=("Courier New", 17, "bold")).pack()
        tk.Label(hdr, text="Mecánica Clásica · RK4 · Unidades Astronómicas",
                 bg=BG, fg=TXT_DIM, font=("Courier New", 9)).pack(pady=(2, 0))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        # Presets
        pf = tk.Frame(self, bg=BG, pady=14)
        pf.pack(fill="x", padx=28)
        tk.Label(pf, text="CASOS PREDEFINIDOS",
                 bg=BG, fg=TXT_DIM,
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        btn_row = tk.Frame(pf, bg=BG)
        btn_row.pack(fill="x", pady=(6, 0))

        for name in PRESETS:
            bg_idle, fg_idle = PRESET_COLORS[name]
            tk.Button(
                btn_row, text=name,
                bg=bg_idle, fg=fg_idle,
                activebackground=fg_idle, activeforeground=BG,
                relief="flat", bd=0, padx=12, pady=7,
                font=("Courier New", 9, "bold"), cursor="hand2",
                command=lambda n=name: self._cargar_preset(n),
            ).pack(side="left", padx=(0, 8))

        self._desc_var = tk.StringVar(
            value="Selecciona un caso predefinido o ingresa manualmente.")
        tk.Label(pf, textvariable=self._desc_var,
                 bg=BG, fg=TXT_DIM, font=("Courier New", 8),
                 justify="left", wraplength=640).pack(anchor="w", pady=(8, 0))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        # Formulario
        form = tk.Frame(self, bg=BG, pady=14, padx=28)
        form.pack(fill="x")
        tk.Label(form, text="CONDICIONES DE BORDE",
                 bg=BG, fg=TXT_DIM,
                 font=("Courier New", 8, "bold")).grid(
                     row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        campos = [
            (1, 0, "r₁  (UA)", "r1",      8, "UA"),
            (1, 2, "θ₁  (°)",  "th1",     8, "°"),
            (2, 0, "r₂  (UA)", "r2",      8, "UA"),
            (2, 2, "θ₂  (°)",  "th2",     8, "°"),
            (3, 0, "M   (M☉)", "M",       8, "M☉"),
            (3, 2, "Pasos RK4","n_pasos", 8, ""),
            (4, 0, "t_total",  "t_total", 8, "años (0=auto)"),
        ]
        for row, col, label, key, width, unit in campos:
            var = tk.StringVar()
            self._vars[key] = var
            tk.Label(form, text=label, bg=BG, fg=TXT,
                     font=("Courier New", 9), anchor="e",
                     width=12).grid(row=row, column=col,
                                    sticky="e", padx=(0, 4), pady=3)
            tk.Entry(form, textvariable=var, width=width,
                     bg=BG3, fg=ACCENT2, insertbackground=ACCENT2,
                     relief="flat", bd=0, font=("Courier New", 10),
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).grid(
                         row=row, column=col+1, sticky="w", padx=(0, 4), pady=3)
            if unit:
                tk.Label(form, text=unit, bg=BG, fg=TXT_DIM,
                         font=("Courier New", 8)).grid(
                             row=row, column=col+2, sticky="w", padx=(2, 14))

        self._vars['vr_neg'] = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form, text="Velocidad radial inicial negativa  (ṙ₀ < 0)",
            variable=self._vars['vr_neg'],
            bg=BG, fg=TXT, activebackground=BG,
            activeforeground=TXT_HI, selectcolor=BG3,
            font=("Courier New", 9), cursor="hand2",
        ).grid(row=5, column=0, columnspan=5, sticky="w", pady=(6, 0))

        # Nombre del caso
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)
        nf = tk.Frame(self, bg=BG, padx=28, pady=10)
        nf.pack(fill="x")
        tk.Label(nf, text="Nombre del caso:", bg=BG, fg=TXT,
                 font=("Courier New", 9)).pack(side="left")
        self._vars['titulo'] = tk.StringVar(value="Mi Órbita")
        tk.Entry(nf, textvariable=self._vars['titulo'],
                 width=36, bg=BG3, fg=ACCENT2,
                 insertbackground=ACCENT2, relief="flat", bd=0,
                 font=("Courier New", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", padx=8)

        # Barra de estado
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)
        self._status_var = tk.StringVar(value="Listo.")
        tk.Label(self, textvariable=self._status_var,
                 bg=BG2, fg=TXT_DIM, font=MONO,
                 anchor="w", pady=5, padx=28).pack(fill="x")

        # Info panel
        self._info_text = tk.Text(
            self, height=5, bg=BG2, fg=TXT_DIM,
            font=MONO, relief="flat", bd=0,
            highlightthickness=0, state="disabled",
            padx=28, pady=10)
        self._info_text.pack(fill="x")

        # Botón simular
        bf = tk.Frame(self, bg=BG, pady=16)
        bf.pack()
        self._run_btn = tk.Button(
            bf, text="▶  SIMULAR ÓRBITA",
            bg=ACCENT, fg=TXT_HI,
            activebackground=ACCENT2, activeforeground=BG,
            relief="flat", bd=0, padx=32, pady=12,
            font=("Courier New", 12, "bold"), cursor="hand2",
            command=self._lanzar_simulacion)
        self._run_btn.pack()

        self._cargar_defaults()

    def _cargar_defaults(self):
        for k, v in {"r1": "1.0", "th1": "0.0", "r2": "1.5", "th2": "90.0",
                     "M": "1.0", "n_pasos": "25000", "t_total": "0.0"}.items():
            self._vars[k].set(v)

    def _cargar_preset(self, nombre):
        p = PRESETS[nombre]
        for k in ('r1', 'th1', 'r2', 'th2', 'M', 'n_pasos', 't_total'):
            self._vars[k].set(str(p[k]))
        self._vars['vr_neg'].set(p['vr_neg'])
        self._vars['titulo'].set(p['titulo'])
        self._desc_var.set(p['desc'])
        self._status_var.set(f"  Preset cargado: {nombre}")

    def _set_info(self, lines):
        self._info_text.config(state="normal")
        self._info_text.delete("1.0", "end")
        self._info_text.insert("end", lines)
        self._info_text.config(state="disabled")

    def _leer_params(self):
        def f(key, minval=None):
            raw = self._vars[key].get().strip()
            if not raw:
                raise ValueError(f"El campo '{key}' está vacío.")
            v = float(raw)
            if minval is not None and v < minval:
                raise ValueError(f"'{key}' debe ser ≥ {minval}.")
            return v
        return dict(
            r1      = f('r1',      minval=0.01),
            th1     = f('th1'),
            r2      = f('r2',      minval=0.01),
            th2     = f('th2'),
            M       = f('M',       minval=0.01),
            n_pasos = int(f('n_pasos', minval=100)),
            t_total = f('t_total', minval=0),
            vr_neg  = bool(self._vars['vr_neg'].get()),
            titulo  = self._vars['titulo'].get().strip() or "Mi Órbita",
        )

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _lanzar_simulacion(self):
        try:
            params = self._leer_params()
        except ValueError as e:
            messagebox.showerror("Error de entrada", str(e))
            return
        self._run_btn.config(state="disabled", text="⏳  Calculando…")
        self._status_var.set("  Calculando órbita…")
        self._set_info("")
        threading.Thread(target=self._hilo_simulacion,
                         args=(params,), daemon=True).start()

    def _hilo_simulacion(self, params):
        error, info = simular(params)
        self.after(0, self._post_simulacion, error, info)

    def _post_simulacion(self, error, info):
        self._run_btn.config(state="normal", text="▶  SIMULAR ÓRBITA")
        if error:
            messagebox.showerror("Error físico", error)
            self._status_var.set("  Error al calcular la órbita.")
            return

        lines = (
            f"  Tipo de órbita : {info['tipo']}\n"
            f"  Excentricidad  : ε = {info['eps']:.6f}\n"
            f"  Mom. angular   : l = {info['l']:.6f}  UA²/yr\n"
            f"  Energía espec. : e = {info['e_esp']:.6f}  UA²/yr²\n"
        )
        if info['colision']:
            lines += f"  ⚠ COLISIÓN detectada en t = {info['t_col']:.4f} yr"
            self._status_var.set("  Simulación completada — ¡colisión detectada!")
        else:
            lines += f"  Error energía  : {info['err_energia']:.4e} %"
            self._status_var.set("  Simulación completada correctamente.")
        self._set_info(lines)


# ============================================================
#  MAIN
# ============================================================

def main():
    app = OrbitApp()
    app.mainloop()


if __name__ == "__main__":
    main()