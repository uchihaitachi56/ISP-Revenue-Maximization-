"""
ISP Revenue Maximization - Price of Simplicity Analysis
Based on: Shakkottai, Srikant, Ozdaglar, Acemoglu (2008)
"The Price of Simplicity", IEEE JSAC

This script replicates Section II examples and extends with a new utility function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize_scalar, minimize
import warnings
warnings.filterwarnings('ignore')

# ─── Color palette ───────────────────────────────────────────────────────────
NAVY   = "#1E3A5F"
TEAL   = "#0D9488"
CORAL  = "#E55C4A"
GOLD   = "#F0A500"
LIGHT  = "#EAF4FB"
GRAY   = "#64748B"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
})

# =============================================================================
# SECTION II REPLICATION  —  Linear Utility Model
# =============================================================================
# Model: User type θ ~ Uniform[0,1], N users
# Utility: U(x, θ) = θ·x  (linear in bandwidth x)
# User maximises: θ·x - p·x  → demands x*=∞ if θ>p, 0 if θ<p
# (We cap at capacity C = 1 for one user)
# ISP uses usage-based pricing p per unit.
#
# With N users and capacity C:
#   Demand fraction at price p:  (1-p) users want positive bandwidth
#   Each demands C/(1-p) in expectation at capacity sharing
#
# Simpler standard approach from paper Section II:
# Two-user example: θ1, θ2 ∈ {θL, θH}
# ===========================================================================

# ─── Example 1: Two-user two-type model (paper's canonical example) ──────────
# Users: type H (θ_H) and type L (θ_L), θ_H > θ_L > 0
# Capacity C = 1 shared equally if both connect
# Usage-based pricing: price p per unit bandwidth
# 
# User type θ connects if θ ≥ p  (net utility θ - p ≥ 0)
# If both connect: each gets C/2 = 0.5 bandwidth
# Revenue from simple pricing (usage-based):
#   Both connect if p ≤ θ_L: R_simple = p * C = p
#   Only H connects if θ_L < p ≤ θ_H: R_simple = p * C = p (gets full capacity)
#
# Optimal (discriminatory) pricing:
#   Offer menu: (x_H, t_H) and (x_L, t_L) subject to IC and IR constraints
#   Under standard Mussa-Rosen mechanism:
#     x_L = 0 (exclude low type), x_H = C
#     t_H = θ_H * C - θ_L * C  [information rent extracted]
#     Actually t_H = θ_H * C in simple exclusion

def analyze_two_user(theta_H=0.8, theta_L=0.3, C=1.0):
    """
    Replicate paper's Section II Example.
    Two-user model with usage-based vs optimal pricing.
    """
    results = {}
    
    # ── Simple (usage-based) pricing ──
    # Optimal simple price maximises R = p * (expected bandwidth sold)
    # Case 1: Set p=θ_L → both buy → R = θ_L * C (each pays θ_L per unit, gets C/2 each)
    R_both = theta_L * C  # both connect at p=θ_L
    # Case 2: Set p=θ_H → only H buys, gets full C
    R_high_only = theta_H * C
    
    R_simple = max(R_both, R_high_only)
    p_simple_opt = theta_L if R_both >= R_high_only else theta_H
    
    # ── Optimal (discriminatory) pricing ──
    # Using second-degree price discrimination (menu of contracts)
    # IR_L: θ_L * x_L - t_L ≥ 0
    # IR_H: θ_H * x_H - t_H ≥ 0
    # IC_L: θ_L * x_L - t_L ≥ θ_L * x_H - t_H
    # IC_H: θ_H * x_H - t_H ≥ θ_H * x_L - t_L
    # 
    # Optimal solution (Myerson/Mussa-Rosen):
    # x_L = 0 (if θ_L < θ_H/2), x_H = C
    # t_L = 0, t_H = θ_H*C - (θ_H - θ_L)*C·(prob(low)/prob(high))
    # With equal probs (1/2 each):
    # t_H = θ_H * C - (θ_H - θ_L) * C = θ_L * C  [if excluding L]
    # OR: serve both: x_L distorted downward
    
    # Serve only H:
    R_opt_excl = theta_H * C  # charge t_H = θ_H*C, x_H = C
    
    # Serve both (no exclusion): virtual valuations
    # With equal probs F(θ)=θ on [θ_L, θ_H], virtual val = θ - (1-F)/f
    # Optimal to include L if virtual val(θ_L) ≥ 0
    # For our discrete 2-type: 
    # R_opt_both = 2*theta_L*C (charge θ_L to both, and θ_H-θ_L extra to H)
    # = θ_L*C + θ_H*C - (θ_H-θ_L)*C... let me compute properly
    # IC_H binding: t_H - t_L = θ_H*(x_H - x_L)
    # IR_L binding: t_L = θ_L * x_L
    # ISP max: t_H + t_L = θ_L*x_L + θ_H*(x_H-x_L) + θ_L*x_L
    #         = θ_H*x_H - (θ_H-2θ_L)*x_L
    # With x_H=C: optimal x_L = C if θ_H < 2θ_L else 0
    
    if theta_H < 2 * theta_L:
        # Include L: x_L = C/2, x_H = C (capacity sharing... simplified)
        # Actually with capacity C shared: x_L + x_H = C
        # Revenue = θ_L*x_L + (θ_H - θ_L + θ_L)*x_L wait let me redo
        # Standard: t_H = θ_H*x_H - θ_H*x_L + θ_L*x_L, t_L = θ_L*x_L
        # With x_H = C, x_L = some fraction
        x_L_opt = C / 2
        x_H_opt = C
        t_L = theta_L * x_L_opt
        t_H = theta_H * x_H_opt - (theta_H - theta_L) * x_L_opt
        R_opt_both = t_L + t_H
    else:
        R_opt_both = 0  # don't serve L
    
    R_opt = max(R_opt_excl, R_opt_both)
    
    # Price of Simplicity
    PoS = R_opt / R_simple if R_simple > 0 else float('inf')
    
    results = {
        'theta_H': theta_H, 'theta_L': theta_L,
        'R_simple': R_simple, 'R_opt': R_opt,
        'PoS': PoS,
        'p_simple_opt': p_simple_opt,
        'R_opt_excl': R_opt_excl,
        'R_opt_both': R_opt_both,
    }
    return results

# ─── PLOT 1: Revenue vs Price for two-user example ──────────────────────────
def plot_revenue_vs_price():
    theta_H, theta_L, C = 0.8, 0.3, 1.0
    
    prices = np.linspace(0.01, 0.99, 400)
    R_usage = []
    
    for p in prices:
        if p <= theta_L:
            # Both connect; each pays p per unit, get C/2
            R_usage.append(p * C)  # total = 2*(p*C/2) = p*C
        elif p <= theta_H:
            # Only H connects, gets full C
            R_usage.append(p * C)
        else:
            R_usage.append(0)
    
    R_usage = np.array(R_usage)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Revenue curve
    ax = axes[0]
    ax.plot(prices, R_usage, color=TEAL, lw=2.5, label='Usage-based revenue R(p)')
    
    # Mark optimal simple price
    idx_opt = np.argmax(R_usage)
    ax.axvline(theta_L, color=GRAY, ls='--', lw=1.2, alpha=0.7)
    ax.axvline(theta_H, color=GRAY, ls='--', lw=1.2, alpha=0.7)
    ax.scatter([prices[idx_opt]], [R_usage[idx_opt]], color=CORAL, s=80, zorder=5,
               label=f'Max R_simple = {R_usage[idx_opt]:.2f}')
    
    # Optimal revenue line
    res = analyze_two_user(theta_H, theta_L, C)
    ax.axhline(res['R_opt'], color=NAVY, ls='-', lw=2, label=f'Optimal R* = {res["R_opt"]:.2f}')
    
    ax.set_xlabel('Usage-based price p')
    ax.set_ylabel('Revenue')
    ax.set_title('Revenue vs Price\n(Two-type model, θ_H=0.8, θ_L=0.3)')
    ax.annotate('θ_L', (theta_L, 0.02), fontsize=9, color=GRAY, ha='center')
    ax.annotate('θ_H', (theta_H, 0.02), fontsize=9, color=GRAY, ha='center')
    ax.legend(frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    
    # Right: PoS vs θ_L for fixed θ_H
    theta_H_fixed = 0.9
    theta_Ls = np.linspace(0.05, theta_H_fixed - 0.05, 200)
    PoS_vals = []
    for tL in theta_Ls:
        r = analyze_two_user(theta_H_fixed, tL)
        PoS_vals.append(r['PoS'])
    
    ax2 = axes[1]
    ax2.plot(theta_Ls, PoS_vals, color=CORAL, lw=2.5)
    ax2.axhline(1.0, color=GRAY, ls='--', lw=1)
    ax2.fill_between(theta_Ls, 1.0, PoS_vals, alpha=0.15, color=CORAL,
                     label='Revenue loss from simplicity')
    ax2.set_xlabel('θ_L (low-type valuation)')
    ax2.set_ylabel('Price of Simplicity (PoS)')
    ax2.set_title(f'PoS vs User Heterogeneity\n(θ_H = {theta_H_fixed} fixed)')
    ax2.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('plot1_two_user.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot 1 saved. PoS = {res['PoS']:.4f}")
    return res

# =============================================================================
# SECTION II REPLICATION — Continuous type distribution
# =============================================================================
# Users: θ ~ Uniform[0,1], capacity C, ISP sets price p per unit
# If θ ≥ p: user demands x = (capacity allocated)
# With large N: each user gets C per unit mass of demand (normalized)
# Revenue from usage-based: R(p) = p * (1-p) * C  [fraction (1-p) connects]
# Optimal discriminatory: R* = ∫_0^1 ψ(θ) dθ where ψ is virtual value
# ψ(θ) = θ - (1-F(θ))/f(θ) = θ - (1-θ) = 2θ - 1
# Serve θ ≥ 1/2: R* = ∫_{1/2}^{1} θ C dθ = C * [θ²/2]_{1/2}^1 = C*(1/2 - 1/8) = 3C/8

def plot_continuous_model():
    """
    Replicate the continuous-type Uniform[0,1] example.
    """
    C = 1.0
    prices = np.linspace(0.001, 0.999, 500)
    
    # Usage-based revenue: R(p) = p*(1-p)*C
    R_usage = prices * (1 - prices) * C
    
    # Flat-rate revenue: charge T for unlimited access
    # User connects if θ*C ≥ T, i.e., θ ≥ T/C
    # R_flat(T) = T * (1 - T/C) * N  (normalized: N=1)
    # With C=1: R_flat(T) = T*(1-T) — same structure
    R_flat = prices * (1 - prices) * C  # Same optimum here
    
    # Optimal revenue
    R_opt = 3 * C / 8  # = 0.375
    
    # Optimal usage price
    p_opt_usage = 0.5  # argmax of p*(1-p)
    R_simple_max = 0.5 * 0.5 * C  # = 0.25
    
    PoS_continuous = R_opt / R_simple_max
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Revenue curves
    ax = axes[0]
    ax.plot(prices, R_usage, color=TEAL, lw=2.5, label='Usage-based R(p) = p(1−p)')
    ax.axhline(R_opt, color=NAVY, lw=2.5, ls='-', label=f'Optimal R* = {R_opt:.3f}')
    ax.scatter([p_opt_usage], [R_simple_max], color=CORAL, s=100, zorder=5,
               label=f'Best simple price: p*=0.5, R={R_simple_max:.3f}')
    
    # Shade revenue gap
    ax.fill_between(prices, R_usage, R_opt,
                    where=(R_usage < R_opt), alpha=0.12, color=CORAL,
                    label=f'Revenue gap (PoS = {PoS_continuous:.3f})')
    
    ax.set_xlabel('Usage price p')
    ax.set_ylabel('Revenue R(p)')
    ax.set_title('Continuous Type Model\nθ ~ Uniform[0,1], Capacity C=1')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    
    # Right: Demand and virtual valuation
    thetas = np.linspace(0, 1, 300)
    virtual_val = 2 * thetas - 1  # ψ(θ) = 2θ-1 for Uniform[0,1]
    
    ax2 = axes[1]
    ax2.plot(thetas, thetas, color=NAVY, lw=2, label='Valuation θ')
    ax2.plot(thetas, virtual_val, color=TEAL, lw=2, ls='--', label='Virtual value ψ(θ)=2θ−1')
    ax2.axhline(0, color=GRAY, lw=1)
    ax2.axvline(0.5, color=CORAL, lw=1.5, ls=':', label='Cutoff θ*=0.5')
    ax2.fill_between(thetas, 0, virtual_val,
                     where=(virtual_val >= 0), alpha=0.15, color=TEAL,
                     label='Served region (ψ≥0)')
    ax2.fill_between(thetas, virtual_val, 0,
                     where=(virtual_val < 0), alpha=0.15, color=CORAL,
                     label='Excluded region (ψ<0)')
    ax2.set_xlabel('User type θ')
    ax2.set_ylabel('Value')
    ax2.set_title('Virtual Valuation & Optimal Mechanism\n(Myerson, 1981)')
    ax2.legend(frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot2_continuous.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot 2 saved. PoS (continuous Uniform) = {PoS_continuous:.4f}")
    return PoS_continuous, R_opt, R_simple_max

# =============================================================================
# NEW UTILITY FUNCTION — Logarithmic (Concave) Utility
# =============================================================================
# U(x, θ) = θ * log(1 + x)  where θ ~ Uniform[0,1], x ∈ [0, C]
#
# User maximises: θ*log(1+x) - p*x  (usage-based, p = price per unit)
# FOC: θ/(1+x) = p  →  x*(θ,p) = max(θ/p - 1, 0)
#      User connects if θ ≥ p  (same threshold)
#
# ISP Revenue from usage-based pricing:
#   R_usage(p) = p * E[x* | θ≥p] * Prob(θ≥p)
#   = p * ∫_p^1 (θ/p - 1) dθ
#   = ∫_p^1 (θ - p) dθ
#   = [(θ-p)²/2]_p^1 = (1-p)²/2
#
# So: R_usage(p) = (1-p)²/2
# Optimal p: dR/dp = -(1-p) = 0 → p=1! But R(1)=0.
# Actually let's redo: R_usage(p) = ∫_p^1 (θ - p) dθ = (1-p)²/2
# dR/dp = -(1-p) < 0 always → monopolist sets p as low as possible?
# This is because of the log shape - we need to include capacity constraint.
#
# With capacity C (rate limit), user demand capped at C:
# x*(θ,p) = min(max(θ/p - 1, 0), C)
# Revenue: R(p) = p * ∫_p^1 min(θ/p - 1, C) dθ
#
# For simplicity, let C=∞ (no capacity) and optimise over transfer scheme
# Optimal mechanism (Myerson):
# ψ(θ) = θ - (1-F(θ))/f(θ) = θ - (1-θ) = 2θ-1  [same for Uniform]
# But allocation: x(θ) maximises θ*log(1+x) - ψ(θ)*... 
#
# More precisely: planner maximises ∫ [ψ(θ)*log(1+x(θ))] dθ
# FOC per type: ψ(θ)/(1+x(θ)) = λ  (shadow price of capacity)
# So x(θ) = max(ψ(θ)/λ - 1, 0)
#
# With capacity C (total): ∫_0^1 x(θ)dθ = C
# Let θ* = 1/2 (virtual val cutoff), served types θ ∈ [1/2, 1]:
# x(θ) = (2θ-1)/λ - 1 for θ ∈ [1/2, 1], x=0 otherwise
# ∫_{1/2}^1 ((2θ-1)/λ - 1) dθ = C
# = [1/λ ∫_{1/2}^1 (2θ-1)dθ] - 1/2 = C
# ∫_{1/2}^1 (2θ-1)dθ = [(θ-1/2)²/... ] = [θ²-θ]_{1/2}^1 = 0 - (-1/4) = 1/4
# 1/(4λ) - 1/2 = C → λ = 1/(4(C+1/2)) = 1/(4C+2)
#
# Optimal revenue:
# R* = ∫_{1/2}^1 θ*log(1+x(θ)) - x(θ)/... (use payment formula)
# R* = ∫_{1/2}^1 ψ(θ)*log(1+x(θ)) dθ  [by revenue equivalence]
# = ∫_{1/2}^1 (2θ-1)*log(1 + (2θ-1)/λ - 1) dθ
# = ∫_{1/2}^1 (2θ-1)*log((2θ-1)/λ) dθ

def analyze_log_utility(C=2.0):
    """
    New utility function: U(x,θ) = θ*log(1+x), θ~Uniform[0,1]
    """
    # ── Usage-based pricing ──
    # R_usage(p) = ∫_p^1 (θ - p) dθ = (1-p)²/2  [without capacity]
    # But with capacity C as individual limit per user:
    # x*(θ,p) = min(θ/p - 1, C), and user connects if θ/p-1 ≥ 0 i.e. θ≥p
    # Also need θ/p - 1 > 0 for positive demand.
    # Break point: θ/p - 1 = C → θ = p(1+C)
    
    prices = np.linspace(0.001, 1.0, 500)
    R_usage_arr = []
    
    for p in prices:
        # Lower threshold: θ ≥ p (participates)
        # Upper threshold: θ ≤ p(1+C) (unconstrained demand)
        theta_lo = p
        theta_hi = min(p * (1 + C), 1.0)
        
        R = 0
        if theta_lo < 1.0:
            if theta_hi > theta_lo:
                # Unconstrained region [theta_lo, theta_hi]: x = θ/p - 1
                # Revenue contribution: ∫_{theta_lo}^{theta_hi} p*(θ/p - 1) dθ
                # = ∫ (θ - p) dθ = [θ²/2 - pθ]_{theta_lo}^{theta_hi}
                def contrib_unconstrained(lo, hi):
                    return (hi**2/2 - p*hi) - (lo**2/2 - p*lo)
                R += contrib_unconstrained(theta_lo, theta_hi)
            
            # Constrained region [theta_hi, 1]: x = C
            if theta_hi < 1.0:
                # Revenue: ∫_{theta_hi}^1 p*C dθ = p*C*(1-theta_hi)
                R += p * C * (1 - theta_hi)
        
        R_usage_arr.append(max(R, 0))
    
    R_usage_arr = np.array(R_usage_arr)
    idx_opt = np.argmax(R_usage_arr)
    p_opt = prices[idx_opt]
    R_simple_max = R_usage_arr[idx_opt]
    
    # ── Optimal mechanism ──
    # ψ(θ) = 2θ - 1, serve θ ≥ 1/2
    # Allocation: x(θ) = min(max(ψ(θ)/λ - 1, 0), C)
    # λ chosen to use capacity: ∫ x(θ)dθ = C*fraction_served (or total budget)
    # Here we maximize total ISP revenue = ∫ t(θ)dθ
    # By Myerson: R* = ∫_{θ*}^1 ψ(θ) * ∂U/∂x * x(θ) dθ... 
    # More directly, optimal transfer: t(θ) = θ*u(x(θ)) - ∫_{θ*}^θ u(x(s)) ds
    # Revenue = ∫_{θ*}^1 [θ*u(x(θ)) - ∫_{θ*}^θ u(x(s))ds] dθ
    # = ∫_{θ*}^1 ψ(θ) * u(x(θ)) dθ  [by integration by parts]
    # u(x) = log(1+x)
    
    # With no capacity constraint, optimal x(θ) for served types:
    # max ∫ ψ(θ)*log(1+x(θ)) dθ subject to ∫ x(θ)dθ ≤ C_total
    # FOC: ψ(θ)/(1+x(θ)) = λ → x(θ) = ψ(θ)/λ - 1 = (2θ-1)/λ - 1
    # for served types (θ ≥ 1/2), set to 0 for θ < 1/2
    
    # Capacity budget: ∫_{1/2}^1 ((2θ-1)/λ - 1) dθ = C_budget
    # [∫(2θ-1)dθ from 1/2 to 1] = 1/4
    # 1/(4λ) - 1/2 = C_budget
    # λ = 1 / (4*(C_budget + 1/2)) = 1/(4C_budget + 2)
    
    # Let C_budget = 1.0 (average capacity per user times 1 user normalized)
    C_budget = 1.0
    lam = 1.0 / (4 * C_budget + 2)
    
    def x_opt(theta):
        if theta < 0.5:
            return 0
        return max((2*theta - 1)/lam - 1, 0)
    
    # Optimal revenue by Myerson formula
    from scipy.integrate import quad
    
    def integrand_opt(theta):
        psi = 2*theta - 1
        x = x_opt(theta)
        return psi * np.log(1 + x)
    
    R_opt, _ = quad(integrand_opt, 0.5, 1.0)
    PoS_log = R_opt / R_simple_max if R_simple_max > 0 else float('inf')
    
    return {
        'prices': prices,
        'R_usage': R_usage_arr,
        'p_opt': p_opt,
        'R_simple_max': R_simple_max,
        'R_opt': R_opt,
        'PoS': PoS_log,
        'lambda': lam,
    }

def plot_log_utility():
    res = analyze_log_utility(C=2.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Left: Revenue vs price
    ax = axes[0]
    ax.plot(res['prices'], res['R_usage'], color=TEAL, lw=2.5,
            label='Usage-based R(p)')
    ax.axhline(res['R_opt'], color=NAVY, lw=2.5, label=f'Optimal R* ≈ {res["R_opt"]:.3f}')
    ax.scatter([res['p_opt']], [res['R_simple_max']], color=CORAL, s=100, zorder=5,
               label=f'Best simple: p*≈{res["p_opt"]:.2f}, R={res["R_simple_max"]:.3f}')
    ax.fill_between(res['prices'], res['R_usage'], res['R_opt'],
                    where=(res['R_usage'] < res['R_opt']), alpha=0.12, color=CORAL)
    ax.set_xlabel('Usage price p')
    ax.set_ylabel('Revenue')
    ax.set_title('Log Utility: U(x,θ)=θ·log(1+x)\nRevenue vs Usage Price')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    
    # Middle: Optimal allocation x(θ)
    thetas = np.linspace(0, 1, 300)
    lam = res['lambda']
    x_vals = np.array([max((2*t-1)/lam - 1, 0) if t >= 0.5 else 0 for t in thetas])
    
    ax2 = axes[1]
    ax2.plot(thetas, x_vals, color=NAVY, lw=2.5, label='Optimal allocation x*(θ)')
    ax2.axvline(0.5, color=CORAL, lw=1.5, ls=':', label='Cutoff θ*=0.5')
    ax2.fill_between(thetas, 0, x_vals, alpha=0.15, color=TEAL)
    ax2.set_xlabel('User type θ')
    ax2.set_ylabel('Bandwidth allocation x*(θ)')
    ax2.set_title('Optimal Mechanism\nBandwidth Allocation per Type')
    ax2.legend(frameon=False)
    
    # Right: Log utility surface
    x_range = np.linspace(0, 5, 100)
    theta_vals = [0.2, 0.5, 0.8, 1.0]
    colors_u = [GRAY, TEAL, CORAL, NAVY]
    
    ax3 = axes[2]
    for theta, col in zip(theta_vals, colors_u):
        ax3.plot(x_range, theta * np.log(1 + x_range), color=col, lw=2,
                 label=f'θ = {theta}')
    ax3.set_xlabel('Bandwidth x')
    ax3.set_ylabel('Utility U(x,θ)')
    ax3.set_title('Logarithmic Utility Function\nU(x,θ) = θ·log(1+x)')
    ax3.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('plot3_log_utility.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot 3 saved. PoS (log utility) = {res['PoS']:.4f}")
    return res

# =============================================================================
# PLOT 4: PoS Comparison across models
# =============================================================================
def plot_pos_comparison():
    """Compare PoS across different models and parameters."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: PoS comparison bar chart
    models = ['Linear\nUniform[0,1]', 'Two-type\n(θ_H=0.8, θ_L=0.3)', 'Log Utility\nUniform[0,1]']
    pos_vals = [3/8/0.25, analyze_two_user(0.8, 0.3)['PoS'], analyze_log_utility()['PoS']]
    colors_bar = [NAVY, TEAL, CORAL]
    
    ax = axes[0]
    bars = ax.bar(models, pos_vals, color=colors_bar, width=0.5, zorder=3)
    ax.axhline(1.0, color=GRAY, ls='--', lw=1.5, label='PoS = 1 (no loss)')
    ax.set_ylabel('Price of Simplicity (PoS = R* / R_simple)')
    ax.set_title('Price of Simplicity Comparison\nAcross Models')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    for bar, val in zip(bars, pos_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylim(0, max(pos_vals)*1.2)
    ax.legend(frameon=False)
    
    # Right: PoS sensitivity (two-type model)
    theta_H_vals = [0.7, 0.8, 0.9]
    theta_Ls = np.linspace(0.05, 0.65, 200)
    
    ax2 = axes[1]
    for tH, col in zip(theta_H_vals, [NAVY, TEAL, CORAL]):
        pos_line = []
        for tL in theta_Ls:
            if tL < tH:
                pos_line.append(analyze_two_user(tH, tL)['PoS'])
            else:
                pos_line.append(np.nan)
        ax2.plot(theta_Ls, pos_line, color=col, lw=2, label=f'θ_H = {tH}')
    
    ax2.axhline(1.0, color=GRAY, ls='--', lw=1)
    ax2.set_xlabel('θ_L (low-type valuation)')
    ax2.set_ylabel('Price of Simplicity')
    ax2.set_title('PoS Sensitivity to User Heterogeneity\n(Two-type model)')
    ax2.legend(frameon=False)
    ax2.set_ylim(0.9, 2.0)
    
    plt.tight_layout()
    plt.savefig('plot4_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 4 (comparison) saved.")
    return pos_vals

# =============================================================================
# PLOT 5: Stackelberg Game Illustration
# =============================================================================
def plot_stackelberg():
    """Illustrate the Stackelberg structure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_facecolor(WHITE)
    
    # ISP box
    isp = mpatches.FancyBboxPatch((0.3, 1.5), 2.2, 1.2,
        boxstyle="round,pad=0.1", fc=NAVY, ec=NAVY, zorder=3)
    ax.add_patch(isp)
    ax.text(1.4, 2.1, "ISP\n(Leader)", ha='center', va='center',
            color=WHITE, fontsize=11, fontweight='bold')
    
    # Users
    for i, (xi, label) in enumerate([(4.5, "User type θ_H"), (7.0, "User type θ_L"), (9.3, "User type θ_M")]):
        col = [TEAL, CORAL, GOLD][i]
        box = mpatches.FancyBboxPatch((xi-1.0, 1.5), 2.0, 1.2,
            boxstyle="round,pad=0.1", fc=col, ec=col, zorder=3)
        ax.add_patch(box)
        ax.text(xi, 2.1, label, ha='center', va='center',
                color=WHITE, fontsize=9, fontweight='bold')
    
    # Arrows: ISP → Users
    for xi in [4.5, 7.0, 9.3]:
        ax.annotate("", xy=(xi-0.9, 2.1), xytext=(2.5, 2.1),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))
    
    # Labels on arrows
    ax.text(3.2, 2.4, "Sets price p\n(or menu)", ha='center', fontsize=8.5, color=GRAY)
    
    # Users → ISP (backward)
    for xi in [4.5, 7.0, 9.3]:
        ax.annotate("", xy=(2.5, 1.9), xytext=(xi-0.9, 1.7),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2, ls='dashed'))
    
    ax.text(3.2, 1.4, "Best response x*(θ,p)", ha='center', fontsize=8.5, color=TEAL)
    
    # Timing labels
    ax.text(1.4, 3.2, "Stage 1", ha='center', fontsize=9, color=NAVY, fontweight='bold')
    ax.text(6.9, 3.2, "Stage 2", ha='center', fontsize=9, color=TEAL, fontweight='bold')
    ax.axvline(3.6, color=GRAY, ls=':', lw=1, ymin=0.3, ymax=0.9)
    
    ax.set_title("Stackelberg Game Structure: ISP Revenue Maximization",
                 fontsize=12, fontweight='bold', color=NAVY, pad=8)
    
    plt.tight_layout()
    plt.savefig('plot5_stackelberg.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 5 (Stackelberg) saved.")

# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ISP Revenue Maximization - Price of Simplicity Analysis")
    print("=" * 60)
    
    print("\n[1] Two-user example (paper replication):")
    res_two = plot_revenue_vs_price()
    print(f"    R_simple = {res_two['R_simple']:.4f}")
    print(f"    R_opt    = {res_two['R_opt']:.4f}")
    print(f"    PoS      = {res_two['PoS']:.4f}")
    
    print("\n[2] Continuous Uniform model (paper replication):")
    PoS_cont, R_opt_c, R_s_c = plot_continuous_model()
    print(f"    R_simple = {R_s_c:.4f}")
    print(f"    R_opt    = {R_opt_c:.4f}")
    print(f"    PoS      = {PoS_cont:.4f}")
    
    print("\n[3] Log utility model (new extension):")
    res_log = plot_log_utility()
    print(f"    R_simple = {res_log['R_simple_max']:.4f}")
    print(f"    R_opt    = {res_log['R_opt']:.4f}")
    print(f"    PoS      = {res_log['PoS']:.4f}")
    
    print("\n[4] Comparison plot:")
    pos_vals = plot_pos_comparison()
    
    print("\n[5] Stackelberg game diagram:")
    plot_stackelberg()
    
    print("\n✓ All plots generated successfully!")
    print("\nSummary Table:")
    print(f"{'Model':<35} {'R_simple':>10} {'R_opt':>10} {'PoS':>8}")
    print("-"*65)
    print(f"{'Linear, two-type (θH=0.8, θL=0.3)':<35} {res_two['R_simple']:>10.4f} {res_two['R_opt']:>10.4f} {res_two['PoS']:>8.4f}")
    print(f"{'Linear, Uniform[0,1] (continuous)':<35} {R_s_c:>10.4f} {R_opt_c:>10.4f} {PoS_cont:>8.4f}")
    print(f"{'Log utility, Uniform[0,1]':<35} {res_log['R_simple_max']:>10.4f} {res_log['R_opt']:>10.4f} {res_log['PoS']:>8.4f}")
