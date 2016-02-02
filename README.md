# World Trade Model
Python code for the World Trade Model (WTM), a linear programming model utilizing input-output economics.  This version includes the Rectangular Choice-of-Technology (RCOT) as well as the option to solve for bilateral trade (WTMBT).

Installation
1) Download the WTM package.

2) Install latest version of Python and/or Python API.  This code has been successfully run using Canopy.

3) Create a Python directory for the WTM package folder (i.e. create a python path for the system you are running)

3) Assure you have a recent version of Excel installed.  This is not necessary to run the WTM, but for ease of loading and managing data inputs and outputs (later versions will have a non-Excel option).

4) Add python packages numpy, pandas, and PuLP.  Numpy and pandas should be included in standard API distributions, and PuLP can be installed using PiP (http://www.coin-or.org/PuLP/main/installing_pulp_at_home.html).

5) Install any additional PuLP solvers you need besides the default CBC solver (GLPK, CPLEX, GUROBI, etc.)

Usage
To solve the 3 region example using the WTM with Rectangular Choice-of-Technology
- Run WTM Version 0.60 in Python or Python API to solve with only resource constraint
- Run WTM Version 0.61 in Python or Python API to solve with both resource constraint and benefit-of-trade constraint
- Run WTMBT Version 0.60 in Python or Python API to solve with resource constrant and bilateral trade constraint
- Run WTMBT Verson 0.61 in Python or Python API to solve with resource constraint, benefit-of-trade constraint, and bilateral trade constraint

Contact nat.springer@gmail.com if interested in using the WTM to solve problems using other datasets.
