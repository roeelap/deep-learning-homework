r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

student_name_1 = 'Roee Lapushin' # string
student_ID_1 = '318875366' # string
student_name_2 = 'Yair Nadler' # string
student_ID_2 = '316387927' # string


# ==============
# Part 1 answers

part1_q1 = r"""
1. The test set allows us to estimate our in-sample error.
False: The test set is used to estimate the out-of-sample error of a model.
In-sample error refers to the error the model makes on the training data. 

2. Any split of the data into two disjoint subsets would constitute an equally useful train-test split.
False: How you split the data into train and test sets can significantly impact the performance estimate 
of your model. For instance, class-imbalanced data requires careful stratification to ensure both the 
train and test sets represent the class distribution accurately.
 
3. The test set should not be used during cross-validation.
True: During cross-validation, the test set should not be used. Cross-validation involves splitting 
the data into multiple subsets (folds) and using each fold as both a training set and a validation 
set in different iterations. Using the test set within cross-validation would introduce data leakage 
and invalidate the cross-validation results.

4. After performing cross-validation, we use the validation-set performance 
of each fold as a proxy for the model's generalization error.
True: In cross-validation, the validation-set performance of each fold provides an estimate of how well 
the model generalizes to new, unseen data. By averaging the performance across all folds, we can get a more 
robust estimate of the model's generalization error compared to a single train-test split, and reduce the 
variance in the performance estimate.
"""

part1_q2 = r"""
No.
Using the test set to choose hyperparameters, including the regularization strength $\lambda$, 
can lead to overfitting to the test set. The purpose of the test set is to provide an unbiased 
evaluation of the model's generalization performance on unseen data. 
If you use the test set multiple times to tune hyperparameters, the model can inadvertently 
learn patterns specific to the test set, reducing its ability to generalize to new, unseen data.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
The selection is arbitrary for the SVM loss because the choice of different $\Delta>0$ doesn't matter on the performance
of the training process. This is because the regularization would provide weights that suit $\Delta$.
"""

part2_q2 = r"""
It looks like the model is learning the weights like a heatmap - adjusting weights to be higher for areas that are
white in a certain image. Images that are stretched out or shifted in a certain direction can produce errors.
"""

part2_q3 = r"""
1. The learning rate is good. We can see in the graph that the accuracy rises quickly at the beginning and then
flattens out. In addition, the loss decreases quickly at the beginning and then flattens out. With too low or too high
learning rates, we would see the accuracy and loss not converge to a good value, or not converge at all.

2. The model is slightly overfitted to the training set. As we can see in the loss graph, the training set's loss is 
improved over time, but the validation set's loss is not.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The ideal pattern would be when the residual graph's mean is as close to 0 as possible, with a low variance.
By looking at the plot, we can see that the trained model is better - the mean stays close to 0, the variance is lower,
and our error decreases. We can see this by comparing the top 5 features' plot with the final plot after the CV,
where the variance got lower. This is clear when comparing the initial plot of the top 5 features with the final
plot after cross-validation, where the variance decreased.
"""

part3_q2 = r"""
1. Yes, it is still linear, because the model is still linear in the features. The polynomial features are
a transformation of the original features, and the model is still linear in these transformed features.

2. Yes, we can fit any non-linear function of the original features with this approach, as the model does not
have to be linear in the original features, only in the features that are input to the model.

3. Applying non-linear transformations to the features can help the model capture more complex patterns in the data,
as the data would move to a higher-dimensional space where it is better separated. the decision boundaries would 
still be a hyperplane in this higher-dimensional space, but it would be a non-linear decision boundary in the original
feature space.
"""

part3_q3 = r"""
**Your answer:**
1. np.logspace returns numbers more evenly spaced out. This is useful when we want to search for hyperparameters
with cross validation on values of lambda that differ greatly, as it would be more efficient.

2. we had 3 different degrees we search over and 20 different lambdas. In total, we had $3 \cdot 20 = 60$ different 
combinations of hyperparameters. Each combination is fitted 3 times - one for each fold in the cross-validation.
Thus, we had $60 \cdot 3 = 180$ fits in total.
"""

# ==============

# ==============
