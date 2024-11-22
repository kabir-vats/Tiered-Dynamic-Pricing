# Introduction

This repository is set up to track progress on a tiered dynamic pricing method that can reach optimal, steady-state pricing for a tiered service based on purchasing behaviors of users.

## The Optimization Problem

This is the optimization problem that the project aims to create a feedback mechanism to solve

#### maximize $\sum_{i = 0}^{B} \left ((p_i-c_i)\sum_{n=0}^{N-1}[i = arg \max_{j = 0}^{B} U(n, j)]\right )$

#### subject  to:

$p_i \geq 0$

$p_i - c_i \geq 0$

$p_0 = 0$

$S - \sum_{i = 0}^{B} \left (c_i\sum_{n=0}^{N-1}[i = arg \max_{j = 0}^{B} U(n, j)]\right )\geq 0$ (this constraint is actually wrong, I don't know how to write it such that we can choose to just sell to half of the available customers)

##### variables:

$P$ vector of prices

##### constants:

$C$ is vector of costs, $c_0 = 0$

$S$ is supply (maximum total cost)

$B$ number of tiers

$N$ is number of total customers

$V$ is a vector of valuation parameters $v_1 \cdots v_N$ decided by a random distribution

#### Utility Function: $U(n,j) = v_n c_1 (\frac{ c_j }{ c_1 })^{1-\frac{1}{ \lambda }} - p_j$

This function incorporates a scaling based on price such that it's anchored at the first tier. So, further tiers' valuation (in the views of the consumer) would be less than proportional to costs, which makes lower tiers more appealing to sell in the highest profit situation.

I think this was the best way to maintain linearity with respect to price while addressing the odd situation I observed through simulations

## Acknowledgements

This work is being completed under the guidance of Prof. Jorge Poveda in the ECE Department of UC San Diego.

### Citations / References / Related Works

The structure of the optimization problem is primarily based on the work explored in [this paper](https://www.sciencedirect.com/science/article/pii/S138912861300340X?casa_token=LJkBgYreLNIAAAAA:94vZKK7_701dl5zZapBUdRnQ3rvvUEmGySAgZ6tB8VojNyvEI5w32wylG8wkBtx7E-Uki3YQ-Io) and [this paper](https://link.springer.com/article/10.1007/s12243-009-0149-3), but I differentiate from these by exploring heterogenous users in a significantly different environment, with a different end goal as well (feedback controls vs optimization). 
