"""
# FM-GQ-M2: Financial Markets — Module 2 Graded Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Financial%20Markets-Derivatives%20%26%20Options-blue" />
  <img src="https://img.shields.io/badge/Option%20Pricing-Black--Scholes%20%26%20Greeks-orange" />
</p>

---

## Overview
This repository documents Module 2 Graded Analytics for Financial Markets (FM). The module focuses on derivative securities analytics, forward and futures pricing, arbitrage-free cost-of-carry relations, the Black-Scholes-Merton European option pricing framework, analytical option sensitivity parameters ($\text{Greeks}$), and implied volatility solving via numerical root-finding.

---

## Part 1: Forwards & Futures Pricing

### Problem 1.1: Cost-of-Carry Forward Pricing

#### Objective
Compute the arbitrage-free forward price $F_0$ and current value $V_0$ of a forward contract written on an underlying asset with spot price $S_0$, risk-free rate $r$, continuous storage cost $u$, and continuous dividend yield $q$ over time to maturity $T$.

#### Mathematical Definition
The continuous cost-of-carry model for forward pricing is:
$$F_0 = S_0 \cdot e^{(r + u - q)T}$$

The value $V_0$ of an existing long forward contract with delivery price $K$ is:
$$V_0 = S_0 \cdot e^{(u-q)T} - K \cdot e^{-rT}$$

---

## Part 2: Black-Scholes-Merton European Option Pricing

### Problem 2.1: BSM Call and Put Valuations

#### Objective
Value European call options ($C_0$) and put options ($P_0$) on stock $S_0$ with strike $K$, risk-free rate $r$, volatility $\sigma$, continuous dividend yield $q$, and time to maturity $T$. Verify the Put-Call Parity relationship.

#### Mathematical Formulas
Black-Scholes-Merton Formula:
$$C_0 = S_0 \cdot e^{-qT} \cdot \mathcal{N}(d_1) - K \cdot e^{-rT} \cdot \mathcal{N}(d_2)$$

$$P_0 = K \cdot e^{-rT} \cdot \mathcal{N}(-d_2) - S_0 \cdot e^{-qT} \cdot \mathcal{N}(-d_1)$$

where:
$$d_1 = \frac{\ln(S_0 / K) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

Put-Call Parity Relationship:
$$C_0 + K \cdot e^{-rT} = P_0 + S_0 \cdot e^{-qT}$$

---

## Part 3: Option Sensitivity Analysis ($\text{Greeks}$) & Implied Volatility

### Problem 3.1: Analytical $\text{Greeks}$ Estimation

#### Objective
Compute key BSM analytical option sensitivity parameters: $\Delta$ ($\text{Delta}$), $\Gamma$ ($\text{Gamma}$), $\mathcal{V}$ ($\text{Vega}$), $\Theta$ ($\text{Theta}$), and $\rho$ ($\text{Rho}$).

#### Mathematical Formulas
$$\Delta_{\text{Call}} = e^{-qT} \cdot \mathcal{N}(d_1), \quad \Delta_{\text{Put}} = -e^{-qT} \cdot \mathcal{N}(-d_1)$$

$$\Gamma = \frac{e^{-qT} \cdot n(d_1)}{S_0 \cdot \sigma \sqrt{T}}, \quad n(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

$$\mathcal{V} = S_0 \cdot e^{-qT} \cdot \sqrt{T} \cdot n(d_1)$$

$$\Theta_{\text{Call}} = -\frac{S_0 \cdot e^{-qT} \cdot n(d_1) \cdot \sigma}{2\sqrt{T}} + q S_0 e^{-qT} \mathcal{N}(d_1) - r K e^{-rT} \mathcal{N}(d_2)$$

---

### Problem 3.2: Implied Volatility Root-Finding

#### Objective
Solve for the implied volatility parameter $\sigma_{\text{imp}}$ given an observed market option price $C_{\text{mkt}}$ using Newton-Raphson root-finding.

#### Mathematical Formula
Solve for $\sigma$ in $f(\sigma) = 0$:
$$f(\sigma) = C_{\text{BSM}}(S_0, K, r, q, T, \sigma) - C_{\text{mkt}} = 0$$

---

## Summary Table of Outputs & Validations

| Task ID | Output Parameter | Metric / Description | Theoretical Target / Formula |
|:---|:---|:---|:---|
| P1.1 | $F_0$ | Cost-of-carry forward price | $S_0 \cdot e^{(r + u - q)T}$ |
| P2.1 | $C_0, P_0$ | BSM European Call and Put values | Closed-form BSM analytic pricing |
| P2.1 | $\text{Parity Error}$ | Put-Call Parity arbitrage check | $\|(C_0 + K e^{-rT}) - (P_0 + S_0 e^{-qT})\| = 0$ |
| P3.1 | $\Delta, \Gamma, \mathcal{V}$ | Analytical BSM Sensitivity Greeks | Partial derivatives $\frac{\partial C}{\partial S}, \frac{\partial^2 C}{\partial S^2}, \frac{\partial C}{\partial \sigma}$ |
| P3.2 | $\sigma_{\text{imp}}$ | Market implied volatility parameter | Root $\sigma^*$ of $C_{\text{BSM}}(\sigma) - C_{\text{mkt}} = 0$ |
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton


# ==========================================
# PART 1: FORWARDS & FUTURES PRICING
# ==========================================

def compute_forward_price_and_value(
    spot_price: float, 
    strike_price: float, 
    risk_free_rate: float, 
    time_to_maturity: float, 
    storage_cost: float = 0.0, 
    dividend_yield: float = 0.0
) -> dict:
    """
    Calculates cost-of-carry forward price F_0 and existing contract value V_0.

    Parameters
    ----------
    spot_price : float
        Current asset spot price S_0.
    strike_price : float
        Delivery price K of long forward contract.
    risk_free_rate : float
        Continuous risk-free rate r.
    time_to_maturity : float
        Time to maturity T in years.
    storage_cost : float
        Continuous storage cost rate u (default = 0.0).
    dividend_yield : float
        Continuous dividend yield q (default = 0.0).

    Returns
    -------
    dict
        Forward price F_0 and current long position value V_0.
    """
    S0, K, r, T, u, q = spot_price, strike_price, risk_free_rate, time_to_maturity, storage_cost, dividend_yield

    # 1. Forward Price: F_0 = S_0 * e^{(r + u - q)T}
    forward_price = S0 * np.exp((r + u - q) * T)

    # 2. Contract Value: V_0 = S_0 * e^{(u-q)T} - K * e^{-rT}
    contract_value = S0 * np.exp((u - q) * T) - K * np.exp(-r * T)

    return {
        'forward_price': float(np.round(forward_price, 4)),
        'contract_value': float(np.round(contract_value, 4))
    }


# ==========================================
# PART 2: BLACK-SCHOLES-MERTON PRICING
# ==========================================

def compute_bsm_option_prices(
    spot_price: float, 
    strike_price: float, 
    risk_free_rate: float, 
    volatility: float, 
    time_to_maturity: float, 
    dividend_yield: float = 0.0
) -> dict:
    """
    Computes European Call and Put option prices under Black-Scholes-Merton model
    and verifies Put-Call Parity.

    Parameters
    ----------
    spot_price : float
        Spot stock price S_0.
    strike_price : float
        Strike price K.
    risk_free_rate : float
        Risk-free rate r.
    volatility : float
        Annualized return volatility \sigma.
    time_to_maturity : float
        Maturity T in years.
    dividend_yield : float
        Continuous dividend yield q.

    Returns
    -------
    dict
        Call price C_0, Put price P_0, d1, d2, and parity error.
    """
    S0, K, r, sigma, T, q = spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, dividend_yield

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # BSM Call & Put pricing
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

    # Put-Call Parity Check: C + K e^{-rT} = P + S_0 e^{-qT}
    lhs_parity = call_price + K * np.exp(-r * T)
    rhs_parity = put_price + S0 * np.exp(-q * T)
    parity_error = np.abs(lhs_parity - rhs_parity)

    return {
        'call_price': float(np.round(call_price, 4)),
        'put_price': float(np.round(put_price, 4)),
        'd1': float(np.round(d1, 6)),
        'd2': float(np.round(d2, 6)),
        'parity_error': float(np.round(parity_error, 8))
    }


# ==========================================
# PART 3: GREEKS & IMPLIED VOLATILITY
# ==========================================

def compute_bsm_greeks(
    spot_price: float, 
    strike_price: float, 
    risk_free_rate: float, 
    volatility: float, 
    time_to_maturity: float, 
    dividend_yield: float = 0.0
) -> dict:
    """
    Computes analytical BSM Option Greeks: Delta, Gamma, Vega, Theta, and Rho.

    Parameters
    ----------
    spot_price : float
        Spot price S_0.
    strike_price : float
        Strike K.
    risk_free_rate : float
        Risk-free rate r.
    volatility : float
        Volatility \sigma.
    time_to_maturity : float
        Maturity T.
    dividend_yield : float
        Dividend yield q.

    Returns
    -------
    dict
        Analytical Greeks for Call and Put options.
    """
    S0, K, r, sigma, T, q = spot_price, strike_price, risk_free_rate, volatility, time_to_maturity, dividend_yield

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 1. Delta: \frac{\partial C}{\partial S}
    delta_call = np.exp(-q * T) * norm.cdf(d1)
    delta_put = -np.exp(-q * T) * norm.cdf(-d1)

    # 2. Gamma: \frac{\partial^2 C}{\partial S^2}
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S0 * sigma * np.sqrt(T))

    # 3. Vega: \frac{\partial C}{\partial \sigma} (per 1% vol change / 0.01)
    vega_raw = S0 * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    vega_1pct = vega_raw * 0.01

    # 4. Theta: \frac{\partial C}{\partial T} (per day / 365)
    theta_call_raw = (- (S0 * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                      + q * S0 * np.exp(-q * T) * norm.cdf(d1) 
                      - r * K * np.exp(-r * T) * norm.cdf(d2))
    theta_call_daily = theta_call_raw / 365.0

    # 5. Rho: \frac{\partial C}{\partial r} (per 1% rate change / 0.01)
    rho_call_raw = K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_call_1pct = rho_call_raw * 0.01

    return {
        'delta_call': float(np.round(delta_call, 4)),
        'delta_put': float(np.round(delta_put, 4)),
        'gamma': float(np.round(gamma, 6)),
        'vega_1pct': float(np.round(vega_1pct, 4)),
        'theta_call_daily': float(np.round(theta_call_daily, 4)),
        'rho_call_1pct': float(np.round(rho_call_1pct, 4))
    }


def solve_implied_volatility(
    market_call_price: float, 
    spot_price: float, 
    strike_price: float, 
    risk_free_rate: float, 
    time_to_maturity: float, 
    dividend_yield: float = 0.0
) -> float:
    """
    Solves for BSM implied volatility using Newton-Raphson root finding.

    Parameters
    ----------
    market_call_price : float
        Observed market price C_mkt.
    spot_price : float
        Spot price S_0.
    strike_price : float
        Strike K.
    risk_free_rate : float
        Risk-free rate r.
    time_to_maturity : float
        Maturity T.
    dividend_yield : float
        Dividend yield q.

    Returns
    -------
    dict
        Solved implied volatility \sigma_imp.
    """
    def price_error(sigma_guess):
        res = compute_bsm_option_prices(spot_price, strike_price, risk_free_rate, sigma_guess, time_to_maturity, dividend_yield)
        return res['call_price'] - market_call_price

    implied_vol = newton(price_error, x0=0.20)

    return float(np.round(implied_vol, 6))


# ==========================================
# EXECUTION TEST & VALIDATION RUNNER
# ==========================================

if __name__ == "__main__":
    print("Executing Financial Markets Module 2 Pipeline...\n")

    # Sample Derivative Inputs: S0=100, K=100, r=5%, sigma=20%, T=1.0 Year, q=1%
    S0, K, r, sigma, T, q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.01

    # Test Part 1: Forwards & Futures
    fwd_res = compute_forward_price_and_value(S0, K, r, T, storage_cost=0.0, dividend_yield=q)
    print("--- Part 1: Forward Contract Pricing ---")
    print(f"Theoretical Forward Price F0: ${fwd_res['forward_price']:,.2f}")
    print(f"Contract Current Value V0   : ${fwd_res['contract_value']:,.2f}")

    # Test Part 2: BSM Pricing & Parity
    bsm_res = compute_bsm_option_prices(S0, K, r, sigma, T, q)
    print("\n--- Part 2: Black-Scholes-Merton Pricing ---")
    print(f"BSM European Call Price     : ${bsm_res['call_price']:,.2f}")
    print(f"BSM European Put Price      : ${bsm_res['put_price']:,.2f}")
    print(f"Put-Call Parity Error Check : {bsm_res['parity_error']}")

    # Test Part 3: Option Greeks & Implied Volatility
    greeks_res = compute_bsm_greeks(S0, K, r, sigma, T, q)
    print("\n--- Part 3: Option Sensitivity Greeks ---")
    print(f"Call Delta (\u0394)               : {greeks_res['delta_call']}")
    print(f"Gamma (\u0393)                    : {greeks_res['gamma']}")
    print(f"Vega (per +1% Vol)          : ${greeks_res['vega_1pct']:,.4f}")
    print(f"Call Theta (Daily)          : ${greeks_res['theta_call_daily']:,.4f}")

    # Solve Implied Volatility from calculated call price
    imp_vol_solved = solve_implied_volatility(bsm_res['call_price'], S0, K, r, T, q)
    print(f"\nSolved Implied Volatility   : {imp_vol_solved * 100:.4f}%")
