# Portfolio Management: Dynamic CVaR Optimization (DCVaR) 
This repository contains the code for the DCVaR model. The model is part of the requirement for a graduate course in HEC Montréal. 

The model consists of two parts: the CVaR optimization and dynamic adjustment bawed on economic regime. They are based on paper from [Rockafellar & Uryasev (2000)](https://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf) and [Kim & Kown (2022)](https://ideas.repec.org/a/pal/assmgt/v24y2023i2d10.1057_s41260-022-00296-8.html). Our model uses inflation and growth indicator to first predict the economic regime for the next period and then perform CVaR optimization using historical data corresponding to that specific regime. It also, in the meanwhile, take into account the client's individual risk tolerance and also the ESG requirment.

After being developed, we compare it to naive 1/N and generic Markowitz mean-variance portfolio. Its performance is desirable but depends on the level of risk tolerance. The slides of the model can be found [here](https://www.dropbox.com/scl/fi/0jtyb94ayen9xm9srp0bq/DCVaR.pdf?rlkey=rgx15vu7li3jx5voi6guy0ac9&dl=0).

The codes is under modification for more accurate numerical computation, reliabiltiy and better interface. The ultimate goal is to illustrate this model that could automatically grab required data and update the portfoilo allaction on a monthly basis. 
