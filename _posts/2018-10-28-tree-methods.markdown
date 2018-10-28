---
layout: post
title:  Seeing the Trees for the Forest | A gentle introduction to tree-based methods Part 2
date:   2018-10-30 17:30:00 -0400
categories: jekyll update
---

### Part 2: Constructing the Tree

In the previous post, we construced the basic building blocks of our decision tree.  To recap, we constructed the following methods:

* `candidate_splits` which takes an X matrix and returns all possible Yes/No questions we can ask of the data (potential split points)
* `vardiance_reduction` which takes an X matrix, y feature, and candidate splits, and returns the variance reduction of each split

Next, we will see how we can recursively apply these methods to partition the X matrix to minimize the variance of y in the resulting "terminal nodes".