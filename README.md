# Introduction

This repository is set up to track progress on a tiered dynamic pricing method that can reach optimal, steady-state pricing for a tiered service based on purchasing behaviors of users. This project was completed as part of a two-quarter honors research capstone for the UC San Diego Jacobs School of Engineering.

## Structure

```
.
├─ figures/                                                     # Figures generated during evaluation
├─ notebooks/                                                   # Original demo notebooks for pricing
├─ pricing/                                                    
│  ├─ dynamic/
│  │  ├─ business.py                                            # Simulate a business agent blind to customer parameters
│  │  ├─ controller.py                                          # Represents a controller of a business. Contains SGD logic.
│  │  ├─ customer.py                                            # Simulate a customer agent that maximizes utility
│  │  ├─ efficient_estimator.py                                 # Use decisions to make estimates
│  │  ├─ estimator.py                                           # Old estimation code
│  │  ├─ experiment/                                            # Code used to generate figures and evaluate different parameter sets
│  ├─ static/
│  │  ├─ experiment/                                            # Code used to evaluate static gradient descent
│  │  ├─ optimize.py                                            # Code used to optimize with gradient descent. Contains gradient calculation logic
│  │  └─ system.py                                              # Code used to simulate the pricing system in static format
│  └─ util/                                                     # Contains visualization code
├─ reports
│  ├─ Final_Tiered_Pricing.pdf                                  # Final research report (READ THIS)
│  └─ first_report.pdf                                          # Preliminary research report
```

## Usage

Please clone the repository and use pip to install requirements.txt

After this, you may use the code in experiment to run certain configurations of the system and observe the results.

## Reports

Please read the `Final_Tiered_Pricing.pdf` report for the full research report from this project. 
