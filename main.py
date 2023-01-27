# import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# import the data
data = pd.read_csv("heart.csv")

"""
Let's start to prepare the data for Logistic Regression Classification. Here, we need to get a small
help from sklearn to split data. Actually, we can succeed in that without using sklearn, but it is a
cumbersome process, and also, this help will not affect our purpose.
"""

# prepare and split the data
y = data["output"].values
x = data.drop("output", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

"""
The shapes of the variables need to be as below:
x_train shape:  (13, 257)
x_test shape:  (13, 46)
y_train shape:  (257,)
y_test shape:  (46,)
"""


"""
When preparing the data, I used the transpose method for the split data. It is just a habit for me. In this method,
each column instead of a row contains data. To understand the codes after this step, you need to have some fundamental 
knowledge about the Logistic Regression method. If you do not have sufficient knowledge, it is going to be so beneficial
to read the article about Logistic Regression in the readme file that is provided in the sources chapter.
"""


# initialize weight and bias
def initializing_weight_and_bias(dimension):
    w = np.full((1, dimension), 0.01)
    b = float()
    return w, b


# sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


"""
In the lines above, we defined a function called "initializing_weight_and_bias" to determine the first
value of weight and bias. Also, thanks to this function, we can determine the shape of weights. After that,
since we need to calculate our z values into the sigmoid function, we defined "sigmoid" function to get y_head
values.
"""


"""
Forward Propagation Steps:

* Calculate z values by multiplying w and x_train values.
* Calculate y_head values by using the sigmoid function. These y_head values are our predictions for current
  weight and bias values.
* Calculate the loss by using log loss. If you do not know anything about log loss, you can take the codes
  directly.
* Calculate the cost function by dividing the loss into the number of elements.

After finishing the forward propagation, our program needs to update our weight and bias. In this way, we will
reach more correct values. To do that, we have to initialize the backward propagation and update our parameters,
but we will update the parameters in the next lines.

Backward Propagation Steps:

* Determine the derivatives of weight and bias values with respect to the given formula. You can also take
  the codes directly if you do not know how to calculate their derivatives.
"""


# forward and backward propagation
def forward_propagation(w, b, x_train, y_train):
    z = np.dot(w, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) + (1-y_train) * np.log(1-y_head)
    cost = np.sum(loss) / x_train.shape[1]
    return y_head, cost


def backward_propagation(x_train, y_train, y_head):
    derivative_weight = np.dot(x_train, (y_head-y_train).T).T / x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train).T / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}
    return gradients


"""
Now, thanks to the parameters we got from the functions above, we can update the weight and bias values. We need
to repeat this process again and again to get the most accurate values. Also, to reach the most accurate values
fast, we need to determine a learning rate parameter.
"""


# update weight and bias
def update(w, b, x_train, y_train, learning_rate, num_iteration):
    cost_list = list()
    cost_list2 = list()
    index = list()
    xticks = list()

    for i in range(num_iteration):
        y_head, cost = forward_propagation(w, b, x_train, y_train)
        gradients = backward_propagation(x_train, y_train, y_head)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        cost_list.append(cost)
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
        if i % 1000 == 0:
            xticks.append(i)
            print("Cost after {}. iteration: {}".format(i, cost))

    parameters = {"weight": w,
                  "bias": b}

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(index, cost_list2, color="red")
    ax.set_xticks(xticks + [xticks[-1] + 1000])
    ax.set_xlabel("Number of Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs. Number of Iteration")
    plt.show()

    return parameters, gradients, cost_list


"""
I wrote some codes above, which are about visualizing the cost function to understand it more easily. Namely, they
do not have any effects on the updating weight and bias process.
"""


"""
After the updating process, we need to predict y_test values by using x_test, and the final weight and bias
values. Our threshold value is 0.5; namely, if a value of test data is higher than 0.5 after the forward
propagation process, the computer will say the output of the test data is 1.
"""


# predict
def predict(w, b, x_test):
    z = sigmoid(np.dot(w, x_test) + b)
    y_pred = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, i] > 0.5:
            y_pred[0, i] = 1

    return y_pred


"""
Now, let's concatenate our functions, and create a Logistic Regression Function.
"""


# logistic regression
def logistic_regression(x_train, x_test, y_train, y_test, learning_rate, num_iteration):
    w, b = initializing_weight_and_bias(x_test.shape[0])
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iteration)
    y_pred = predict(parameters["weight"], parameters["bias"], x_test)
    score = 100 - np.mean(np.abs(y_pred - y_test)) * 100
    print("Test accuracy without sklearn: {:.4f} %".format(score))


logistic_regression(x_train, x_test, y_train, y_test, 21e-7, 10000)


"""
As seen, our Logistic Regression function predicted the values with 80.43% accuracy and printed the
detailed figures about the analysis. The learning_rate and num_iteration variables need to be tuned by
hand. Therefore, you can reach higher values by changing them. Now, let us compare our result with
the result which is going to be found by sklearn.
"""


log_reg = LogisticRegression()
log_reg.fit(x_train.T, y_train.reshape(-1, 1))
score = log_reg.score(x_test.T, y_test.reshape(-1, 1))
print("Test accuracy with sklearn: {:.4f} %".format(score * 100))
