import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number of features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = torch.normal(mean=0, std=weight_std, size=(n_features, n_classes))

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.
        class_scores = x @ self.weights
        y_pred = class_scores.argmax(dim=1)

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.
        acc = (y == y_pred).float().mean()

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.
            loss = 0
            total_correct = 0
            count = 0

            # evaluate on the entire training set
            for x_train, y_train in dl_train:
                y_pred, class_scores = self.predict(x_train)
                batch_size = y_train.shape[0]
                total_correct += self.evaluate_accuracy(y_train, y_pred) * batch_size
                loss += loss_fn.loss(x_train, y_train, class_scores, y_pred) + (weight_decay / 2) * torch.norm(self.weights, 2)
                count += batch_size

                grad = loss_fn.grad()
                grad += weight_decay * self.weights
                self.weights -= learn_rate * grad

            train_res[0].append(total_correct / count)
            train_res[1].append(loss / len(dl_valid))

            # evaluate on the validation set
            loss = 0
            total_correct = 0
            count = 0
            for x_valid, y_valid in dl_valid:
                y_pred, class_scores = self.predict(x_valid)
                batch_size = y_valid.shape[0]
                total_correct += self.evaluate_accuracy(y_valid, y_pred) * batch_size
                loss += loss_fn.loss(x_valid, y_valid, class_scores, y_pred)
                count += batch_size

            valid_res[0].append(total_correct / count)
            valid_res[1].append(loss / len(dl_valid))
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        if has_bias:
            w_images = self.weights[1:, torch.arange(self.weights.shape[1])]
        else:
            w_images = self.weights

        w_images = w_images.T.view(w_images.shape[1], *img_shape)

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.

    hp["weight_std"] = 0.01
    hp["learn_rate"] = 0.05
    hp["weight_decay"] = 0.008

    return hp
