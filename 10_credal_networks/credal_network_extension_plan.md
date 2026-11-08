# PGMPY Credal Network Extension

## Parameterization

Add a new Conditional Probability Distribution type for Credal Probability Distributions. We will first do the discrete/tabular version of a credal set conditional parameterization, and then if needed the continuous, Linear Gaussian Credal parameterization will follow.

### CPDs
I will add a new `credal_tabular.py` under the ```cpds/``` directory. This class will inherit the `BaseParameter` class.

### Base Parameter
I will extend the base parameter class by adding a variable type "discrete-credal".
I will have to alter/extend the ```fit()``` method to be compatible with discrete credal parameters.
I will have to implement a new ```predict_proba()``` to be compatible with discrete credal sets.
The other methods needed will follow.

### Distributions
I do not believe I need to implement the distribution class for Credal Networks as an initial feature.

### Concerns:
The compatibility with skpro, sklearn etc. Credal Sets and Credal Networks are niche and these libraries will probably not have support.

## Parameter Estimator

### Background
My Masters work focuses on using Credal Networks in a Reinforcement Learning loop that follows a Markov Decision Process. In standard model-free RL, the parameters of the policy function or Q-function are updated at every time step, and are learnt over many runs of the RL loop. Therefore, my parameter estimation implementation has to be fine-grained enough to update the parameters at each time step, rather than only offering a traditional ```.fit()``` implementation.

### Algorithm
To learn the parameters (i.e. the credal sets) of the Credal Network, I plan on using the Imprecise Dirichlet Model (IDM) [Walley, 1996](https://doi.org/10.1111/j.2517-6161.1996.tb02065.x). Unlike Expectation Maximisation and Maximum Likelihood estimation, the IDM does not optimise the parameters towards some goal or desired distribution, or to maximise some likelihood. It provides the mathematical mechanism for turning counts into credal sets: each parameter is represented by a count, and the lower and upper probability bounds are derived directly from those counts.

The IDM is inherently incremental. Observing a data point increases the relevant counts, and the credal sets are then recomputed from the updated counts. Processing a whole dataset is therefore just the same single-observation update applied repeatedly. This means the estimator can be built around one primitive, with the batch case built on top of it.

### Implementation

I will add a class called `DiscreteCredalParameterEstimator` that will inherit `BaseParameterEstimator`. Next I will add the `discrete_credal_idm.py` estimator class, which will expose two methods:

1) **`update(observation, weight=1.0)`**: a single-observation update. Given one data point, it increments the counts of the parameters that the observation touches by `weight`, and recomputes the affected credal sets. The default weight of 1 gives the standard IDM count update from Walley and the papers that follow it.
2) **`fit(data)`**: the batch method. It follows the standard IDM process by iterating through the dataset and applying `update` to each row, so that at the end you are left with credal sets based purely on counts. This is the general-purpose entry point for anyone wanting to learn a credal network from data.

The `weight` argument is what makes the single-step update useful beyond standard counting, because the caller decides how much each observation contributes. In my Masters work, this is how I will combine the IDM with the REINFORCE algorithm: at each time step I call `update` with a weight derived from the return, so that the credal sets are scaled towards higher-return actions. The REINFORCE loop, the return computation and the choice of weight all live in my thesis code, and the library only needs to provide the weighted count update. The estimator itself remains a general credal network parameter estimator with no knowledge of reinforcement learning.

## Inference

For inference I plan to use the Credal Variable Elimination [Mauá et al.](https://www.sciencedirect.com/science/article/pii/S0888613X12000904). I plan to do this by creating a new CredalVariableElimination class based on the current VariableElimination class.

Similar to discrete Bayesian networks, this will result in posterior probabilities, however in this case the posterior probability distributions are credal sets.

## Decision Making

In standard Bayesian Networks, you would use Maximum a Posteriori (MAP) estimation in order to find the prediction that maximises the likelihood of your intended situation. However, with credal networks, we are dealing with a set of probability distributions (i.e. credal sets) at each node, therefore we cannot simply use MAP inference to make a decision.

There are several methods for making decisions under imprecise probabilities, see the paper ["Decision making under uncertainty using imprecise probabilities" by Matthias C.M. Troffaes](https://www.sciencedirect.com/science/article/pii/S0888613X06000387)

For this I plan to make a decision making class under the inference folder where these different decision making methods can be housed and used in conjunction with CredalVariableElimination to get a prediction.

The criteria in Troffaes compare decisions by their expected gain, so a decision making method needs to be told what each outcome is worth. The decision making class will therefore take two inputs: the posterior credal set returned by CredalVariableElimination, and a utility (gain) function over the values of the queried variable, supplied by the user. [Default if no utility is supplied: to be decided, for example a 0-1 utility that rewards a single target value.] The class will support the following criteria: Γ-maximin, Γ-maximax, interval dominance, maximality and E-admissibility. Γ-maximin and Γ-maximax return a single decision. Interval dominance, maximality and E-admissibility can return a set of admissible decisions, and the class will return that full set. Choosing a single decision from the set (for example by random tie-breaking) is left to the caller, since this is application specific. In my Masters work, I will break ties uniformly at random in my own code.

## Credal Network Structure

There are 3 ways in which to go about the structure of a credal network.

1) We use traditional structure learning algorithms and scores. This would mean that only the parameters being credal sets makes it a credal network. (This may be a bit naive, but it makes sense)
2) We manually specify the network structure for credal networks. (This is the approach I am planning to take for my Masters Research)
3) Continue doing research into structure learning algorithms for credal networks that use imprecise probabilities to learn the structure of the network.

I am inclined to opt for method 2, as it is the easiest implementation given how niche credal networks are.

## Factors

I would also need to extend the classes inside the factors files, so that CredalVariableElimination can calculate the factors for the credal sets.
