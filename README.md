# SynthGenerator
The repository contains some ongoing experiments with synthetic data generation for AutoML benchmarking.
It is planned to provide a user-friendly API for synthetic datasets generation that can be used in  vanilla ML-models and AutoML algorithms benchmarking.

R&D is maintained by [ITMO-NSS-Team](https://github.com/ITMO-NSS-team) as a part of [FEDOT](https://github.com/nccr-itmo/FEDOT) AutoML framework development.

## Motivation
Imagine that you have successfully built and trained a ML-pipeline that showed a good cross-validation score. However, several questions might have occurred:                        
+ Will my model return robust predictions based on the new data?                                                                          
+ What if my ML-pipeline is too complex and can be easily replaced by a single model (for instance, by a tuned xgboost)?
+ What are the worst cases for my ML-pipeline?

These questions will lead you to use different algorithms and frameworks for synthetic data generation to evaluate the advantages and disadvantages of the solution.
However, most of the open-source frameworks are very limited and only provide low-level parameters for configuration.
Several state-of-the-art experimental algorithms exist but it is still an open problem.

### *What is the state-of-the-art?*
+ [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) - synthetic datasets generator
+ [MDCGenpy](https://github.com/CN-TU/mdcgenpy) - Multidimensional Dataset for Clustering Generator
+ [HAWK](https://github.com/sea-shunned/hawks) - an experimental framework for clustering datasets generation based on evolutionary algorithm
+ [EDO](https://github.com/daffidwilde/edo) - a python library for generating artificial datasets through evolution

### *What about AutoML?*
In the AutoML case, it is also an open problem since most of the frameworks are usually tested using only a predefined list of datasets for benchmarking.

Speaking about the [composite models](https://itmo-nss-team.github.io/FEDOT.Docs/autolearning/composite) that can be built by AutoML frameworks (for instance, using FEDOT), a specific generator of synthetic datasets is also needed for performance benchmarking. 

Therefore, the goal of the research is to implement a useful instrument for its purposes.
