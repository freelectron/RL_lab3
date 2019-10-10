# Reinforcement Learning: Reproducibility Lab

_There are different types of experience replay, e.g. prioritised experience
replay and hindsight experience replay. Compare two or more types of
experience replay. Does the `winner' depend on the type of environment?_

The current plan is to compare the performance of normal experience replay,
prioritized experience replay (PER), and hindsight experience replay (HER)
using the DQN algorithm. The goal is to test these three on two sets of
environments that show the element of **stochasticity** (TBD) of rewards around the
starting state, to determine the efficacy of the three types of experience
replay against this specific problem.