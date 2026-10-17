
# FM-GQ-M1: Financial Markets — Module 1 Graded Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Financial%20Markets-Fixed%20Income-blue" />
  <img src="https://img.shields.io/badge/Bond%20Pricing-Yields%20%26%20Duration-orange" />
</p>

---

## Overview
This repository documents Module 1 Graded Analytics for Financial Markets (FM). The module covers core fixed income analytics, cash flow discounting, bond pricing, Yield to Maturity ($\text{YTM}$) root-finding, Macaulay and Modified Duration, Bond Convexity, and Money Market discount yield conversions.

---

## Part 1: Bond Pricing & Yield to Maturity (YTM)

### Problem 1.1: Fixed-Coupon Bond Pricing

#### Objective
Compute the clean price $P_0$ of a fixed-coupon bond given face value $F$, annual coupon rate $C$, compounding frequency $k$, annual yield to maturity $y$, and remaining maturity $T$ years.

#### Mathematical Definition
The full price $P_0$ of a bond paying periodic coupon $c = \frac{C \cdot F}{k}$ over $n = k \cdot T$ periods is:
$$P_0 = \sum_{t=1}^{n} \frac{c}{\left(1 + \frac{y}{k}\right)^t} + \frac{F}{\left(1 + \frac{y}{k}\right)^n}$$

---

### Problem 1.2: Yield to Maturity (YTM) Numerical Solving

#### Objective
Solve for the annual yield to maturity $y^*$ given an observed market bond price $P_{\text{mkt}}$ using Newton-Raphson numerical root-finding.

#### Mathematical Formula
Solve for $y$ in $f(y) = 0$:
$$f(y) = \sum_{t=1}^{n} \frac{c}{\left(1 + \frac{y}{k}\right)^t} + \frac{F}{\left(1 + \frac{y}{k}\right)^n} - P_{\text{mkt}} = 0$$

---

## Part 2: Sensitivity Analysis — Duration & Convexity

### Problem 2.1: Macaulay & Modified Duration

#### Objective
Compute Macaulay Duration $D_{\text{mac}}$ (weighted average time until cash flow receipt) and Modified Duration $D_{\text{mod}}$ (price sensitivity to yield changes).

#### Mathematical Formulas
Macaulay Duration:
$$D_{\text{mac}} = \frac{1}{P_0} \sum_{t=1}^{n} \frac{t}{k} \cdot \frac{c_t}{\left(1 + \frac{y}{k}\right)^t}$$

Modified Duration:
$$D_{\text{mod}} = \frac{D_{\text{mac}}}{1 + \frac{y}{k}}$$

First-Order Price Approximation:
$$\frac{\Delta P}{P_0} \approx -D_{\text{mod}} \cdot \Delta y$$

---

### Problem 2.2: Bond Convexity & Taylor Series Price Estimation

#### Objective
Calculate bond convexity $C_x$ to account for curvature in the price-yield relationship and estimate price changes under large yield shifts $\Delta y$.

#### Mathematical Formulas
Convexity Measure $C_x$:
$$C_x = \frac{1}{P_0 \cdot \left(1 + \frac{y}{k}\right)^2} \sum_{t=1}^{n} \frac{t(t+1)}{k^2} \cdot \frac{c_t}{\left(1 + \frac{y}{k}\right)^t}$$

Second-Order Price Percentage Change Approximation:
$$\frac{\Delta P}{P_0} \approx -D_{\text{mod}} \cdot \Delta y + \frac{1}{2} C_x \cdot (\Delta y)^2$$

---

## Part 3: Money Market Rates & Day-Count Conventions

### Problem 3.1: Bank Discount Yield vs. Bond Equivalent Yield

#### Objective
Convert Money Market Bank Discount Yield ($y_{\text{bd}}$) to Bond Equivalent Yield ($y_{\text{bey}}$) under Actual/365 and Actual/360 day-count conventions.

#### Mathematical Formulas
Bank Discount Yield:
$$y_{\text{bd}} = \frac{F - P}{F} \cdot \frac{360}{t_{\text{mat}}}$$

Bond Equivalent Yield ($y_{\text{bey}}$):
$$y_{\text{bey}} = \frac{F - P}{P} \cdot \frac{365}{t_{\text{mat}}} = \frac{365 \cdot y_{\text{bd}}}{360 - (y_{\text{bd}} \cdot t_{\text{mat}})}$$

---

## Summary Table of Outputs & Validations

| Task ID | Output Parameter | Metric / Description | Theoretical Target / Formula |
|:---|:---|:---|:---|
| P1.1 | `bond_price` | Present value of bond cash flows | $\sum_{t=1}^n \frac{c_t}{(1+y/k)^t} + \frac{F}{(1+y/k)^n}$ |
| P1.2 | `ytm_solved` | Annualized Yield to Maturity | Root $y^*$ of $P(y) - P_{\text{mkt}} = 0$ |
| P2.1 | `macaulay_duration` | Cash flow weighted average time | $\frac{1}{P_0} \sum_{t=1}^n \frac{t}{k} \text{PV}(c_t)$ |
| P2.1 | `modified_duration` | Price sensitivity per 1% yield change | $\frac{D_{\text{mac}}}{1 + y/k}$ |
| P2.2 | `convexity` | Price-yield curvature measure | Second derivative term $\frac{1}{P_0}\frac{d^2P}{dy^2}$ |
| P3.1 | `bey_yield` | Money market Bond Equivalent Yield | $\frac{365 \cdot y_{\text{bd}}}{360 - y_{\text{bd}} t_{\text{mat}}}$ |
"""

import numpy as np
import pandas as pd
from scipy.optimize import newton


# ==========================================
# PART 1: BOND PRICING & YTM SOLVING
# ==========================================

def compute_bond_price(
    face_value: float, 
    coupon_rate: float, 
    ytm: float, 
    years_to_maturity: float, 
    frequency: int = 2
) -> float:
    """
    Computes the clean present value price of a fixed-coupon bond.

    Parameters
    ----------
    face_value : float
        Par value of the bond F.
    coupon_rate : float
        Annual coupon rate C (decimal).
    ytm : float
        Annual yield to maturity y (decimal).
    years_to_maturity : float
        Time to maturity T in years.
    frequency : int
        Coupon payments per year k (default = 2 for semi-annual).

    Returns
    -------
    float
        Present value bond price P_0.
    """
    n_periods = int(years_to_maturity * frequency)
    coupon_pmt = (coupon_rate * face_value) / frequency
    periodic_ytm = ytm / frequency

    # Periods array 1..n
    periods = np.arange(1, n_periods + 1)
    
    # Cash flows: coupons + par at maturity
    cash_flows = np.full(n_periods, coupon_pmt)
    cash_flows[-1] += face_value

    # Present value discounting
    discount_factors = (1.0 + periodic_ytm) ** (-periods)
    bond_price = float(np.sum(cash_flows * discount_factors))

    return float(np.round(bond_price, 4))


def solve_bond_ytm(
    market_price: float, 
    face_value: float, 
    coupon_rate: float, 
    years_to_maturity: float, 
    frequency: int = 2
) -> float:
    """
    Solves for annual Yield to Maturity (YTM) using Newton-Raphson method.

    Parameters
    ----------
    market_price : float
        Observed market price P_mkt.
    face_value : float
        Par value F.
    coupon_rate : float
        Annual coupon rate C.
    years_to_maturity : float
        Years until maturity T.
    frequency : int
        Payments per year k.

    Returns
    -------
    float
        Solved annual YTM y*.
    """
    def price_error(y):
        return compute_bond_price(face_value, coupon_rate, y, years_to_maturity, frequency) - market_price

    # Newton-Raphson root finding starting at coupon rate guess
    ytm_solution = newton(price_error, x0=coupon_rate)

    return float(np.round(ytm_solution, 6))


# ==========================================
# PART 2: DURATION & CONVEXITY ANALYTICS
# ==========================================

def compute_duration_and_convexity(
    face_value: float, 
    coupon_rate: float, 
    ytm: float, 
    years_to_maturity: float, 
    frequency: int = 2
) -> dict:
    """
    Calculates Macaulay Duration, Modified Duration, and Convexity.

    Parameters
    ----------
    face_value : float
        Par value F.
    coupon_rate : float
        Annual coupon rate C.
    ytm : float
        Annual yield to maturity y.
    years_to_maturity : float
        Maturity T in years.
    frequency : int
        Payment frequency k.

    Returns
    -------
    dict
        Macaulay duration, Modified duration, and Convexity.
    """
    n_periods = int(years_to_maturity * frequency)
    coupon_pmt = (coupon_rate * face_value) / frequency
    periodic_ytm = ytm / frequency

    periods = np.arange(1, n_periods + 1)
    time_years = periods / frequency

    cash_flows = np.full(n_periods, coupon_pmt)
    cash_flows[-1] += face_value

    discount_factors = (1.0 + periodic_ytm) ** (-periods)
    pv_cash_flows = cash_flows * discount_factors
    bond_price = np.sum(pv_cash_flows)

    # 1. Macaulay Duration: \frac{1}{P_0} \sum t_i \cdot \text{PV}(CF_i)
    mac_duration = np.sum(time_years * pv_cash_flows) / bond_price

    # 2. Modified Duration: \frac{D_{\text{mac}}}{1 + y/k}
    mod_duration = mac_duration / (1.0 + periodic_ytm)

    # 3. Convexity: \frac{1}{P_0 (1+y/k)^2} \sum \frac{t(t+1)}{k^2} \text{PV}(CF_i)
    convexity = np.sum(periods * (periods + 1) * pv_cash_flows) / (bond_price * (frequency**2) * ((1.0 + periodic_ytm)**2))

    return {
        'bond_price': float(np.round(bond_price, 4)),
        'macaulay_duration_years': float(np.round(mac_duration, 4)),
        'modified_duration_years': float(np.round(mod_duration, 4)),
        'convexity': float(np.round(convexity, 4))
    }


def estimate_price_change_taylor(
    modified_duration: float, 
    convexity: float, 
    delta_y: float
) -> dict:
    """
    Estimates percentage price change using 1st order (Duration) 
    and 2nd order (Duration + Convexity) Taylor expansion.

    Parameters
    ----------
    modified_duration : float
        Modified duration D_mod.
    convexity : float
        Bond convexity C_x.
    delta_y : float
        Change in yield \Delta y (e.g. +0.01 for +100 bps).

    Returns
    -------
    dict
        1st order duration estimate and 2nd order duration+convexity estimate.
    """
    # 1st order: -D_mod * \Delta y
    pct_change_duration = -modified_duration * delta_y

    # 2nd order: -D_mod * \Delta y + 0.5 * C_x * (\Delta y)^2
    pct_change_convexity = pct_change_duration + 0.5 * convexity * (delta_y ** 2)

    return {
        'delta_y_bps': float(np.round(delta_y * 10000.0, 1)),
        'duration_approx_pct': float(np.round(pct_change_duration * 100.0, 4)),
        'duration_convexity_approx_pct': float(np.round(pct_change_convexity * 100.0, 4))
    }


# ==========================================
# PART 3: MONEY MARKET CONVERSIONS
# ==========================================

def convert_discount_to_bond_equivalent_yield(
    bank_discount_yield: float, 
    days_to_maturity: int
) -> float:
    """
    Converts a Money Market Bank Discount Yield (y_bd) to Bond Equivalent Yield (y_bey).

    Parameters
    ----------
    bank_discount_yield : float
        Annualized bank discount yield y_bd (decimal).
    days_to_maturity : int
        Days remaining until maturity t_mat.

    Returns
    -------
    float
        Bond Equivalent Yield y_bey (decimal).
    """
    # y_bey = (365 * y_bd) / (360 - y_bd * t_mat)
    bey = (365.0 * bank_discount_yield) / (360.0 - (bank_discount_yield * days_to_maturity))

    return float(np.round(bey, 6))


# ==========================================
# EXECUTION TEST & VALIDATION RUNNER
# ==========================================

if __name__ == "__main__":
    print("Executing Financial Markets Module 1 Pipeline...\n")

    # Sample Bond Inputs: $1,000 Par, 6% Coupon, 5.0% YTM, 10 Years, Semi-Annual
    F, C, y_init, T, k = 1000.0, 0.06, 0.05, 10.0, 2

    # Test Part 1: Bond Pricing & YTM Solving
    price_calc = compute_bond_price(F, C, y_init, T, k)
    ytm_solved = solve_bond_ytm(price_calc, F, C, T, k)

    print("--- Part 1: Fixed Income Pricing & YTM ---")
    print(f"Calculated Bond Price   : ${price_calc:,.2f}")
    print(f"Solved YTM from Price   : {ytm_solved * 100:.4f}%")

    # Test Part 2: Duration, Convexity & Sensitivity
    sens_res = compute_duration_and_convexity(F, C, y_init, T, k)
    print("\n--- Part 2: Duration & Convexity ---")
    print(f"Macaulay Duration       : {sens_res['macaulay_duration_years']} Years")
    print(f"Modified Duration       : {sens_res['modified_duration_years']} Years")
    print(f"Convexity               : {sens_res['convexity']}")

    # Price sensitivity for +200 bps yield shock (+0.02)
    shock_res = estimate_price_change_taylor(
        sens_res['modified_duration_years'], 
        sens_res['convexity'], 
        delta_y=0.02
    )
    print(f"\nYield Shock (+200 bps) Duration-Only Approx : {shock_res['duration_approx_pct']}%")
    print(f"Yield Shock (+200 bps) Duration+Convexity  : {shock_res['duration_convexity_approx_pct']}%")

    # Test Part 3: Money Market Rate Conversion (180-day T-Bill with 4.0% Discount Yield)
    bey_res = convert_discount_to_bond_equivalent_yield(bank_discount_yield=0.04, days_to_maturity=180)
    print("\n--- Part 3: Money Market Yield Conversion ---")
    print(f"Bank Discount Yield (180d) : 4.0000%")
    print(f"Bond Equivalent Yield (BEY): {bey_res * 100:.4f}%")
