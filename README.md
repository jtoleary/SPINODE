# Stochastic Physics-Informed Neural Ordinary Differential Equations (SPINODE)

Stochastic differential equations (SDEs) are used to describe a wide variety of complex stochastic dynamical systems. Learning the hidden physics within SDEs is crucial for unraveling fundamental understanding of these systems’ stochastic and nonlinear behavior. We propose a flexible and scalable framework for training artificial neural networks to learn constitutive equations that represent hidden physics within SDEs. The proposed stochastic physics-informed neural ordinary differential equation framework (SPINODE) propagates stochasticity through the known structure of the SDE (i.e., the known physics) to yield a set of deterministic ODEs that describe the time evolution of statistical moments of the stochastic states. SPINODE then uses ODE solvers to predict moment trajectories. SPINODE learns neural network representations of the hidden physics by matching the predicted moments to those estimated from data. Recent advances in automatic differentiation and mini-batch gradient descent with adjoint sensitivity are leveraged to establish the unknown parameters of the neural networks. We demonstrate SPINODE on three benchmark in-silico case studies and analyze the framework's numerical robustness and stability. SPINODE provides a promising new direction for systematically unraveling the hidden physics of multivariate stochastic dynamical systems with multiplicative noise.

# Other
Please note that the file "CSA.zip" contains example data and results for a colloidal self-assembly system case study with an exogenous input.

# Help
Please direct all questions to jared.oleary@berkeley.edu

# Citation



