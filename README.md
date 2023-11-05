# interplanetary-shocks
A collection of code, drawing on Speasy and Trotta code, to analyse interplanetaru shocks.

A note on the different folders and files:
> 230601-Val-Code: Val's initial code on which I based my own;
> 231016-Project-Code: The code I submitted for my project;
> 231023-Post-Project: The code I submitted for my dissertation;
> 231101-Simple-Program: Program to retrieve satellite data using Speasy and plot it;
> 231104-Testing-Additive: Program to retrieve data to test the additive hypothesis;
> 231105-Clean-Program: A clean version of my program that extracts the appropiate data, carries out RH analysis, and plots the data.

The files used in my program:
> shock_analysis.ipynb: The main Jupyter notebook to retrieve, analyse, and plot the plasma data
> data_functions.py: The functions to retrieve and filter the plasma data;
> calc_functions.py: The functions used to analyse the plasma data;
> rankine_functions.py: The functions used to analyse the RH quantities of the data;
> SerPyShock.py: The functions from the Trotta paper.

The only file that needs to be changed is the shock_analysis.ipynb file, changing the satellite time, spacecraft, and method.
