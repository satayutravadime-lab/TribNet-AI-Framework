# TribNet-AI Framework

**Physics-informed machine learning framework for multi-objective optimisation of self-lubricating nano-coatings**

## Overview

TribNet-AI integrates experimental tribological data, molecular dynamics simulations, 
and first-principles DFT calculations to optimise nano-coating compositions. The framework 
uses stacked ensemble learning with multi-objective Bayesian optimisation and closed-loop 
active learning.

## Key Features

- Multi-source data ingestion (experimental tribometry, MD simulations, DFT)
- Physics-informed ensemble prediction (XGBoost + Gaussian Process + PINN)
- Multi-objective Bayesian optimisation (EHVI acquisition function)
- Closed-loop active learning with uncertainty sampling

## System Requirements

- Python 3.11
- NVIDIA GPU (recommended: A100 or equivalent)
- Operating system: Linux / Windows / macOS

## Installation

```bash
git clone https://github.com/satayutravadime-lab/TribNet-AI-Framework.git
cd TribNet-AI-Framework
pip install -r requirements.txt
