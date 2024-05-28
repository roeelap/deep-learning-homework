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
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1. The shape of the Jacobian tensor for the output with respect to the input would be 
 $N \times out features \times N \times in features$. So it will be $64 \times 512 \times 64 \times 1024$.

2. The Jacobian is indeed sparse. specifically, the partial derivative is zero for all the elements that are not in the diagonal.

3. We do not need to materialize the entire Jacobian. The gradient with respect to $W$ can be computed directly using
 the chain rule. The gradient is computed as: $\delta X = \delta Y^T W$.
 This is due to the fact that: $\delta X_{jl} = \sum_{i=1}^{512} \delta y_{ji} w_{il}$.
"""

part1_q2 = r"""
**Your answer:**
Back-propagation itself is not strictly required, but an efficient method for computing gradients is essential for 
training neural networks with decent-based optimization. It efficiently computes the gradients of the loss function
with respect to the weights, which are necessary for updating the weights during training. Without back-propagation,
gradient computation would be computationally expensive and impractical for large networks.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 5
    lr = 0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.25
    lr_vanilla = 0.04
    lr_momentum = 0.025
    lr_rmsprop = 0.0001
    reg = 0.005
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0024
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. The graphs roughly align with our expectations based on the dropout results. Without dropout, we see overfitting, 
 which leads to poor performance on the test set. With medium dropout, we get the best results, avoid 
 overfitting and improving test performance. However, with high dropout, where most neurons are dropped, the results 
 become highly random, varying based on the seed.

2. As previously mentioned, high dropout introduces randomness, evident in the non-monotonic behavior of
 the graph. Conversely, with no dropout, overfitting occurs as expected.
"""

part2_q2 = r"""
**Your answer:**
This is possible. It can occur when the errors we made had a large margin of error, 
resulting in significant losses. However, overall, we made fewer mistakes in the epoch, leading to an increased test 
score. This pattern can continue for several epochs, where the gradient tries to minimize the score of the erroneous 
points, which may negatively affect a different subset of results, but still improve the overall performance.
"""

part2_q3 = r"""
**Your answer:**
1. Gradient Descent optimizes parameters by iteratively descending along the negative gradient of a cost function. 
 Back-Propagation, on the other hand, is a method for efficiently computing gradients in neural networks by propagating 
 errors backward from the output layer to the input layer.

2. Gradient Descent computes the gradient of the cost function using the entire training set at each iteration, 
 making it computationally expensive and less suitable for large datasets. Stochastic Gradient Descent, on the other 
 hand, updates the parameters using a single training example or a small batch of examples at each iteration. 
 This makes SGD faster and more scalable for large datasets, although it introduces more noise into the optimization process.
 
3. SGD is favored in deep learning due to several reasons. Firstly, its scalability allows it to handle large datasets 
 that may not fit into memory if processed using GD. Secondly, the faster update frequency of SGD results in quicker 
 iterations during training, speeding up the overall learning process. Thirdly, the noise introduced by SGD's stochastic 
 updates can help escape local minima and improve the generalization ability of the model. 

4. This method is equivilent. Let's denote the loss function as $L(\theta)$ where $\theta$ are the model parameters. 
 The gradient of the loss function for a batch $B_i$ can be written as: $ g_i = \frac{\partial L_{B_i}(\theta)}{\partial \theta} $
 If we split the data into $m$ disjoint batches, the total loss is the sum of the losses of all batches: $ L(\theta) = \sum_{i=1}^m L_{B_i}(\theta) $
 The gradient of the total loss is the sum of the gradients of the individual batch losses: $ g = \frac{\partial L(\theta)}{\partial \theta} = \sum_{i=1}^m \frac{\partial L_{B_i}(\theta)}{\partial \theta} = \sum_{i=1}^m g_i $
 The out-of-memory error likely occurs because during the implementation, the losses for each batch are accumulated 
 before performing the backward pass. Accumulating the losses requires storing the computational graph for each batch 
 until the backward pass is done. If the dataset is large, the memory required to store all these computational graphs 
 can exceed the available memory.
"""

part2_q4 = r"""
**Your answer:**
1. In forward mode AD, we can reduce memory complexity by computing the gradients sequentially without storing 
 intermediate values, maintaining $\mathcal{O}(n)$ computation cost and $\mathcal{O}(1)$ memory complexity.
 Similarly, in backward mode AD, we can reduce memory complexity by computing gradients in reverse order, 
 avoiding storing intermediate values and maintaining $\mathcal{O}(n)$ computation cost and $\mathcal{O}(1)$ memory complexity.
 
2. Yes. Forward mode AD computes gradients sequentially along the graph without storing intermediate values, 
 while backward mode AD computes gradients in reverse order, both maintaining $\mathcal{O}(n)$ computation cost and $\mathcal{O}(1)$ memory complexity.

3. For deep architectures, memory-efficient AD techniques are crucial. With reduced memory complexity, 
 backpropagation can handle deep networks with large numbers of layers and parameters without exhausting memory 
 resources, facilitating faster and more scalable training.
"""
# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 256
    activation = "relu"
    out_activation = "softmax"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. The model does not have high optimization error as the loss curve for the training set does not show significant 
 fluctuations during training, and it is decreasing over time. In addition, the accuracy curve fot the training set
 does reach a high value.

2. The model does not have a high generalization error as there is no significant gap between the training and
 validation loss and accuracy curves.

3. The model has a small approximation error as the model is not able to approximate the underlying pattern
 in the data in a perfect manner. This is evident by observing the decision boundary plot.
"""

part3_q2 = r"""
**Your answer:**
As we have full knowledge on the data generation process, we can expect the FPR and FNR of the validation set to be
pretty similar to the training set. This is because the validation set has the exact same distribution as the training set.
"""

part3_q3 = r"""
**Your answer:**
1. In this case, we would like to minimize the false positive rate - meaning lowering the threshold. 
 We would like to minimize the number of healthy people that are classified as sick in order to save money and time on
 unnecessary tests. If a sick person is classified as healthy, we don't need to worry as much, as the disease would probably not kill them.
 
2. In this case, we would like to minimize the false negative rate - meaning raising the threshold. 
 We would like to minimize the number of sick people that are classified as healthy in order to save lives. 
 If a healthy person is classified as sick, we can always run more tests to make sure they are healthy, but if a sick 
 person is classified as healthy, they might not get the treatment they need, and die.
"""


part3_q4 = r"""
1. With fixed depth, varying the width affected the complexity of the decision boundaries. Narrower widths led
 to simpler decision boundaries, while wider widths captured more intricate patterns. As we can see in the plots,
 when the width was 8, the model was at its best accuracy for the test set. When the width was 32, the model 
 overfit a little bit.

2. With fixed width, varying the depth influenced the model's ability to represent hierarchical features 
 and complex relationships. As seen in the plots, the deeper the model, the more complex the decision boundaries, 
 and the better the model's performance.

3. As  we can see, the model with width 8 and depth 4 achieved better results for the validation set, but the model
 with width 32 and depth 1 achieved better results for the test set. 
 This is because the model with width 8 and depth 4 is more complex and can capture more intricate patterns, but it
 is also more prone to overfitting. The model with width 32 and depth 1 is simpler and less prone to overfitting.
 The differences though are not very significant.
 
4. Adjusting the threshold affects the model's sensitivity and specificity, influencing the trade-off between false 
 positives and false negatives. Optimal threshold selection based on the validation set aims to balance these factors.
 Improvements in validation set performance due to threshold selection did not always translated directly to test 
 set performance. Overfitting to the validation set is possible if the threshold is fine-tuned excessively. 
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.02
    weight_decay = 0.01
    momentum = 0.5
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. The parameter count per layer (including bias) is represented by $C = \text{out} \cdot (\text{size}^2 \cdot \text{in} + 1)$. 
 For the given example, the parameter count is calculated as follows: $C_1 = 2 \cdot 256 \cdot (9 \cdot 256 + 1) = 1,180,160$ parameters. 
 In the bottleneck block scenario, incorporating 2 convolutional layers of size 3x3, the parameter count 
 becomes $C_2 = 64 \cdot (1 \cdot 256 + 1) + 64 \cdot (9 \cdot 64 + 1) + 256 \cdot (1 \cdot 64 + 1) = 70,016$ parameters. 
 Further, with 2 3x3 convolutions, it becomes $C_3 = 64 \cdot (1 \cdot 256 + 1) + 2 \cdot 64 \cdot (9 \cdot 64 + 1) + 256 \cdot (1 \cdot 64 + 1) = 106,944$ 
 parameters, significantly fewer than the original residual block's parameter count.
 
2. It's evident that the residual block would incur a higher performance cost with regards to FLOPS due to executing 
 actions in a higher-dimensional space with a 3x3 convolution, as opposed to the residual bottleneck block operating in
 a lower-dimensional space.

3. This aspect depends on whether we implement 2 3x3 convolutions on the bottleneck block. 
 If we do, the bottleneck block can spatially combine in a similar manner to the conventional residual block,
 resulting in the same receptive field in the H x W dimensions. By merging channels in the bottleneck block's input and 
 redistributing them in the output, we achieve cross-feature map combination, which is more challenging to achieve in 
 the regular residual block.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""