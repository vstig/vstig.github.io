---
layout: post
title:  Seeing the Trees for the Forest | A gentle introduction to tree-based methods Part 2
date:   2018-11-01 17:30:00 -0400
categories: jekyll update
published: false
---

### Part 2: Constructing the Tree

In the previous post, we construced the basic building blocks of our decision tree.  To recap, we constructed the following methods:

* `candidate_splits` which takes an X matrix and returns all possible Yes/No questions we can ask of the data (potential split points)
* `vardiance_reduction` which takes an X matrix, y feature, and candidate splits, and returns the variance reduction (of y) for each split

Next, we will see how we can recursively apply these methods to partition the X matrix to minimize the variance of y in the resulting "terminal nodes".  Putting these pieces together, we can construct a _Decision Tree Regressor_ to predict y for new, unseen data.

Note that this is not intended is an optmized Decision Tree implementation by any means, but rather one that hopefully trades off efficiency for conceptual clarity.

# Table of contents
1. [Creating the Node Object](#node)
2. [Creating the Tree Object](#tree)
    1. [Stopping Parameters](#stopping)
    2. [Model Fitting](#fitting)
    3. [Model Predicting](#predict)
3. [Inspecting the Tree](#inspect)

### <a name="node"></a> Creating the Node Object 
First we will consider what information we want each split point,
or "decision node", to contain.  In the last post, we saw that the essential information at a split point is which feature to split on, and the value to split at.  We can begin to sketch out this object as follows:

{% highlight python %}
class Node(object):
    def __init__(self):
        self.feature_split = None
        self.feature_value = None
{% endhighlight %}

In addition to this information, we also want to track the splits that occur recursively on the each of the two resulting subsets.  Finally, we will include any other information that may be useful or interesting to reference (e.g. the index or id of all samples at that node).  At the very least, this should include the average value of the variable we are trying to predict.  Here I am also including an indicator for whether we are at a terminal node (mildly useful but not necessary, as this is implicit in the absense or presence of children).

Now we have:

{% highlight python %}
class Node(object):
    def __init__(self):
        self.feature_split = None
        self.feature_value = None

        self.left_child = None
        self.right_child = None

        self.avg = None
        
        self.is_terminal = False
{% endhighlight %}

This sets up the basic scaffolding, but for this object to be useful, we also want to define a method that can take an X matrix and feature y, and split the data so as to maximize the reduction of y's variance in the resulting partitions of X.

Most of the heavy lifting here was accomplished in the first post, where we developed the function to select the best split.  We can define a `split_data` method as follows:

{% highlight python %}
def split_data(self, X, y):
    self.avg = X[y].mean()
    
    candidate_splits = get_candidate_splits(X, y)
    self.feature_split, self.feature_value = get_best_split(X, candidate_splits, y)
    
    X_i, X_j = split_data(X, self.feature_split, self.feature_value)
    
    return X_i, X_j
{% endhighlight %}

Note that in addition to returing the resulting data splits `X_i` and `X_j`, we also update _most_ of the information initialized to None on object creation.  We now have enough information to reconstruct the split on new data, and can return the average y value of all training data at that point.

However, we still have not set the `left_child` and `right_child` attributes.  This coordination between nodes will be orchestrated by the Tree object, which we will develop next.

Here is an example usage of this node object on the [_Wages and Education of Young Males_](https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Males.html) _R_ dataset:

{% highlight python %}
n = Node()
print(n)

X_i, X_j = n.split_data(male_wages, 'wage')
print(n)
{% endhighlight %}

```
==> feature: None
split value: None
terminal node: False
Avg y: None

feature: school
split value: 12.0
terminal node: False
Avg y: 1.6491471906705262
```

As we can see, this node, when given the entire dataset _X_, split on the feature ***school***, and whether the observation has more or less than 12 years of schooling.

Lastly, I am going to define a utility method on the `Node` class that takes a new sample _X_ and returns whether it is routed to the left or right subtree.  This will help when I am trying to understand the decision path for a particular observation:

{% highlight python %}
def which_branch(self, X):
    """
    determines whether new observation X goes to left or right branch
    if split feature is integer: left means < value
    if split feature is bool: left means X[feature] is False
    if split feature is category: left means X[feature] != value
    """
    if self.feature_split is None:
        return "Node has not generated split"
    if np.issubdtype(type(X[self.feature_split]), np.number):
        if X[self.feature_split] < self.feature_value:
            return 'left'
        else:
            return 'right'
    elif np.issubdtype(type(X[self.feature_split]), np.bool_):
        if X[self.feature_split]:
            return 'right'
        else:
            return 'left'
    else:
        if X[self.feature_split] != self.feature_value:
            return 'left'
        else:
            return 'right'
{% endhighlight %}


### <a name="tree"></a>Creating the Tree Object
#### <a name="stopping"></a> Stopping Parameters
To construct the Tree, we first need a brief overview of the common parameters that dictate its construction.  The most important are the rules that control when a tree stops splitting.  This is how we can optimize the tree for predictive performance _and_ generalizability.  At one extreme, suppose we continue to split until there is just a single sample in each node (or all the samples are identical in X, and thus no further splits are possible).  In this case, we can model even highly non-linear relationships and are able to fit the dataset _exactly_ (low train error), but this would likely result in poor generalization to new data (high test error).  At the other extreme, we could stop splitting after the first split (or before we even split at all!), and essentially predict the population mean for all inputs.

Training a decision tree often involves finding the optimal balance between these two extremes, where we can model complex relationships, but limit the model complexity so as to not overlearn noise or nuances of the test data ("overfitting").

Here we will introduce two common "stopping parameters":

* minimum samples per leaf _n_: don't split the data if either of the resulting groups would have less than _n_ samples
* maximum depth _n_: don't split the data if it has already ben split

You can imagine other stopping rules (e.g. some minimum threshold of variance reduction for split), but we will consider these two in our implementation.  Now we can begin to sketch out the structure of our `Tree` object:

{% highlight python %}
class Tree(object):
    def __init__(self, depth=0, min_samples_leaf=5, max_depth=10):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        
        self.depth = depth
        
        self.root_node = None
        self.is_fitted = False
{% endhighlight %}

Now we need to define some methods to make the Tree useful.  Using common convention for model functions, we will define the following:

* `fit`: train a model given data `X` and feature name to predict `y`
* `predict`: traverse the tree to return predicted `y` for new data

Let's first present some pseudocode for model fitting:

#### <a name="fitting"></a> Model Fitting

```
def fit(X, y):
	Is the tree already fit? -> Stop

	Initialize the root node
	Split X at root node, into X_i, X_j

	### Check the stopping rules
	Are we at max depth? -> Stop, flag root node as terminal
	Are X_i or X_j less than minimum samples per leaf? -> Stop, flag root node as terminal

	### Recursively build sub-trees
	Initialize left, right children as Trees w/ depth = current depth + 1

	Fit left child w/ X_i, y
	Fit right child w/ X_j, y
```

We can write this as follows:

{% highlight python %}
def fit(self, X, y):
    if self.is_fitted:
        return "Tree has aleady been fit"
    self.root_node = Node()
    X_i, X_j = self.root_node.split_data(X, y)
    if (
        self.depth+1 <= self.max_depth and # have we reached maximum depth?
        X_i.shape[0]>=self.min_samples_leaf and # would split make left tree have less than min_samples_leaf?
        X_j.shape[0]>=self.min_samples_leaf # would split make right tree have less than min_samples_leaf?
    ):
        self.root_node.left_child = Tree(depth=self.depth+1,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_depth=self.max_depth)
        self.root_node.left_child.fit(X_i, y)

        self.root_node.right_child = Tree(depth=self.depth+1,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_depth=self.max_depth)
        self.root_node.right_child.fit(X_j, y)
    else:
        self.root_node.is_terminal = True

    self.is_fitted = True
{% endhighlight %}

#### <a name="predict"></a> Model Predictions

Once a Tree has been fit, we can predict `y` for a new data point `X_i` by traversing the tree from root note to terminal node according to the split points at each node and the feature values of `X_i`.  Once we reach a terminal node, we can look up the `Node.avg` value to return as our prediction for `X_i`.

{% highlight python %}
def predict(self, X):
    if not self.is_fitted:
        return "Tree not fit yet"
    else:
        node = self.root_node
        while not node.is_terminal:
            if which_branch(X, node.feature_split, node.feature_value) == 'left':
                node = node.left_child.root_node
            else:
                node = node.right_child.root_node

        return node.avg 
{% endhighlight %}

### <a name="inspect"></a>Inspecting the Tree

Now let's inspect the structure of this tree, and gain some insight as to how a particular prediction is made.  As we saw at the end of the Node section above, we would expect the first split on the data to be whether school is great or less than 12 years.  We can confirm this by looking training our newly created Tree object and inspecting it's root node as follows:

{% highlight python %}
t = Tree(min_samples_leaf=5, max_depth=4)
t.fit(male_wages, 'wage')

print(t.root_node)
{% endhighlight %}

```
feature: school
split value: 12.0
terminal node: False
```

We can also traverse our tree in print out the decisions at each split, for example to find the leaf with the maximum average y value:

{% highlight python %}
ct = 0
node = t.root_node
while not node.is_terminal:
    print('Split {}'.format(ct))
    print(node)
    print('Average y: {:.2f}'.format(node.avg))
    if node.left_child.root_node.avg > node.right_child.root_node.avg:
        print("left branch")
        node = node.left_child.root_node
    else:
        node = node.right_child.root_node
        print("right branch")
    ct += 1
    print('\n')
    
print('Split {}'.format(ct))
print(node)
print('Average y: {:.2f}'.format(node.avg))
{% endhighlight %}


```
Split 0
feature: school
split value: 12.0
terminal node: False
Average y: 1.65
right branch

Split 1
feature: year
split value: 1984.0
terminal node: False
Average y: 1.74
right branch

Split 2
feature: school
split value: 13.0
terminal node: False
Average y: 1.86
right branch

Split 3
feature: industry
split value: Trade
terminal node: False
Average y: 2.01
left branch

Split 4
feature: industry
split value: Agricultural
terminal node: True
Average y: 2.05
```

We can also compare our tree to the tree produced by scikit's DecisionTreeRegressor with the same parameters (min_samples_leaf = 5, max_depth=4):

<img align="center" style="width:400%;height:400%;margin:0px 10px" src="/assets/tree.png" alt="Decision Tree Structure"/>

Note that _value_ in the scikit tree corresponds to the _Average y_ we print out from traversing our `Tree` object.  We also could have very easily stored the variance (or "mse") at each node if we wanted.  As you can see, the path we printed out (and associated y values) is the same sequence of splits leading to the maximal value in the tree shown above.

Finally, we can further validate our `Tree` implementation by training multiple Trees with different parameter settings, and calculating a measure of prediction prediction accuracy.  For this, We will train and test on the same dataset, so we would expect increasing prediction accuracy as we increase the model flexibility (decrease min_samples_leaf and/or increase max_depth).

Specifically, we are going to set max_depth to infinity and iterate over min_samples_leaf of 1, 5, 10, 20, and 50 samples.  For each model, we will calculate the r-squared metric, which is a number generally between 0 and 1 that describes the amount of variance in ***y*** that is explained by the model.  A value of 0 means we are not explaining any of the variance, and a value of 1 indicates we are perfectly modeling the data.  We expect to see the decrease as we increase the minimum samples per leaf.

{% highlight python %}
from sklearn.metrics import r2_score, mean_squared_error
results = []

for s in [1, 5, 10, 20, 50]:
    t = Tree(min_samples_leaf=s, max_depth=float('inf'))
    t.fit(male_wages, 'wage')

    preds = [t.predict(row) for ix, row in male_wages.iterrows()]
    results.append({'min_samples_leaf': s,
                    'r-squared': r2_score(male_wages['wage'], preds),
                    'mean_squared_error': mean_squared_error(male_wages['wage'], preds)})
    
pd.DataFrame(results).plot(x='min_samples_leaf', y=['mean_squared_error', 'r-squared'],
                          title='Decision Tree training error as a function of sample size',
                          figsize=(12, 8))
{% endhighlight %}

<img align="center" style="width:400%;height:400%;margin:0px 10px" src="/assets/training_error.png" alt="Decision Tree Structure"/>

I'll close this out with one final dive into the decision rules learned by our tree.  In particular, suppose we want to know what features of our wages dataset are most indicative of high wages.  Let's look at the story that is told by traversing our most complex tree, trained with `min_samples_leaf = 1`, for the terminal node with maximum average wages.  This will be an extension of the rules learned on the shallow tree above.

```
Split 0
feature: school
split value: 12.0
Average y: 1.65
right branch

    Split 1
    feature: year
    split value: 1984.0
    Average y: 1.74
    right branch

        Split 2
        feature: school
        split value: 13.0
        Average y: 1.86
        right branch

            Split 3
            feature: industry
            split value: Trade
            Average y: 2.01
            left branch

        Split 4
        feature: industry
        split value: Agricultural
        Average y: 2.05
        left branch

    Split 5
    feature: year
    split value: 1985
    Average y: 2.07
    right branch

        Split 6
        feature: industry
        split value: Finance
        Average y: 2.11
        right branch

            Split 7
            feature: residence
            split value: north_east
            Average y: 2.34
            right branch

                Split 8
                feature: school
                split value: 14
                Average y: 2.69
                right branch

                    Split 9
                    feature: exper
                    split value: 7.5
                    Average y: 2.81
                    left branch

                Split 10
                feature: year
                split value: 1986
                Average y: 2.99
                right branch

                    Split 11
                    feature: None
                    split value: None
                    terminal node: True
                    Average y: 3.08
```

These rules say that, in this dataset, we see highest wages for individuals in Finance in the North East that over 14 years of school and less than 8 years of experience, in years after 1986.  Most of these make sense, although I wouldn't have expected _less_ than 8 years of experience to be associated with _higher_ wages, for some subset of the population (split 9).  However this could be a fluke of overfitting the particular dataset and a small sample size at that point in the tree.  

We should be wary of reading much into these rules (in particular, the splits at deeper parts of the tree will be based on very few observations when we set min_samples_leaf to 1), or into overly complex decision tree in general.  However, I do believe that poking around is heplful for understanding how trees work (and when and why they may not) and exploring general trends in the data.

In the next post and final post of this series, I hope to extend the ideas one step further to Random Forests, where we can create an _ensemble_ of trees to reduce the risk of overfitting while still allowing us to model highly complex, non-linear relationships.

The code developed in this post and the last can be explored in more detail at the repo [here](https://github.com/vstig/MyTree).
