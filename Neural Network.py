from Mnist import load_mnist
from matplotlib import pyplot as plt
import numpy as np
import random
import pickle

random.seed(1)
np.random.seed(1)
(train_images, train_labels), (test_images, test_labels) = load_mnist()
print(f"Old training shape: {train_images.shape}")
print(f"Old test shape: {test_images.shape}")

train_images = np.expand_dims(train_images,axis=1)
test_images = np.expand_dims(test_images, axis=1)
print(f"New training shape: {train_images.shape}")
print(f"New test shape: {test_images.shape}")

print(train_labels.shape)
print(test_labels.shape)

num_rows = 4
num_cols = 5
num_samples = num_rows * num_cols
random_indices = np.random.randint(0, len(train_images), num_samples)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5 * num_cols, 2 * num_rows))

for i, idx in enumerate(random_indices):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    image = train_images[idx]
    label = train_labels[idx]
    image_to_plot = image.squeeze()
    ax.imshow(image_to_plot,cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


def samples_take(n_train=50000, n_test=1000):
    random.seed(1)
    np.random.seed(1)
    train_idx = np.random.choice(train_images.shape[0], n_train,replace=False)
    trX = train_images[train_idx]
    trY = train_labels[train_idx]
    test_idx = np.random.choice(test_images.shape[0], n_test, replace=False)
    tsX = test_images[test_idx]
    tsY = test_labels[test_idx]
    trX = trX.reshape(-1, 28 * 28).T
    trY = trY.reshape(1, -1)
    tsX = tsX.reshape(-1, 28 * 28).T
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY
trX, trY, tsX, tsY = samples_take()
print('trX.shape: ', trX.shape)
print('trY.shape: ', trY.shape)
print('tsX.shape: ', tsX.shape)
print('tsY.shape: ', tsY.shape)

def relu(Z):
    cache = {}
    A = np.maximum(0, Z)
    cache = {"Z": Z}
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z <= 0] = 0
    return dZ

def linear(Z):
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache

def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A=exp_Z/np.sum(exp_Z,axis=0,keepdims=True)
    cache = {}
    cache["A"] = A
    if Y.size == 0:
        loss = []
    else:
        m = Y.shape[1]
        log_probs = -np.log(A[Y, range(m)])
        loss = np.sum(log_probs) / m
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    A = cache["A"]
    n, m = A.shape
    Y_one_hot = np.zeros_like(A)
    Y_one_hot[Y, range(m)] = 1
    dZ = (A - Y_one_hot) / m
    return dZ

def dropout(A, drop_prob, mode='train'):
    mask = None
    if drop_prob == 0:
        cache = (drop_prob, mode, mask)
        return A, cache
    prob_keep = 1 - drop_prob
    if mode == 'train':
        mask = (np.random.rand(*A.shape) < prob_keep) / prob_keep
        A = A * mask
    elif mode != 'test':
        raise ValueError("Mode value not set correctly, set it to 'train' or 'test'")
    cache = (drop_prob, mode,mask)
    return A, cache

def dropout_der(dA_in, cache):
    dA_out = None
    drop_out, mode, mask = cache
    if not drop_out:
        return dA_in
    if mode == 'train':
        dA_out = dA_in * mask
    elif mode == 'test':
        dA_out = dA_in
    return dA_out


def batchnorm(A, beta, gamma):
    if beta.size == 0 or gamma.size == 0:
        cache = {}
        return A, cache
    epsilon = 1e-5
    mu = np.mean(A, axis=1,keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    std_inv = 1. / np.sqrt(var + epsilon)
    A_norm = (A - mu) * std_inv
    Anorm = gamma * A_norm + beta
    cache = {
        'A': A,
        'mu': mu,
        'var': var,
        'std_inv': std_inv,
        'A_norm': A_norm,
        'gamma': gamma,
        'beta': beta,
        'epsilon': epsilon}
    return Anorm, cache


def batchnorm_der(dA_in, cache):
    if not cache:
        dbeta = []
        dgamma = []
        return dA_in, dbeta, dgamma

    A = cache['A']  # (n,m)
    mu = cache['mu']  # (n,1)
    var = cache['var']  # (n,1)
    std_inv = cache['std_inv']  # (n,1)
    std_inv = np.clip(std_inv, -1e3,1e3)
    A_norm = cache['A_norm']  # (n,m)
    gamma = cache['gamma']  # (n,1)
    epsilon = cache['epsilon']
    m = A.shape[1]  # batch size

    dbeta = np.sum(dA_in, axis=1,keepdims=True)
    dgamma = np.sum(dA_in * A_norm, axis=1, keepdims=True)  # (n,1)
    dA_norm = dA_in * gamma  # (n,m)

    term1 = dA_norm
    term2 = np.sum(dA_norm, axis=1, keepdims=True) / m
    term3 = A_norm * np.sum(dA_norm * A_norm, axis=1, keepdims=True) / m
    dA_out = std_inv * (term1 - term2 - term3)

    return dA_out, dbeta, dgamma

def initialize_network(net_dims, act_list, drop_prob_list):
    net_dims_len = len(net_dims)
    parameters = {}
    parameters['numLayers'] = net_dims_len - 1
    for l in range(net_dims_len-1):
        parameters["act"+str(l+1)] = act_list[l]
        parameters["dropout"+str(l+1)] = drop_prob_list[l]
        W = np.random.randn(net_dims[l+1], net_dims[l]) * np.sqrt(2/net_dims[l])
        b = np.zeros((net_dims[l+1], 1))
        parameters["W"+str(l+1)] = W
        parameters["b"+str(l+1)] = b
    return parameters


def initialize_velocity(parameters, apply_momentum=True):
    L = parameters['numLayers']
    parameters['apply_momentum'] = apply_momentum

    for l in range(L):
        if apply_momentum:
            parameters["VdW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
            parameters["Vdb" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
            parameters["GdW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
            parameters["Gdb" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return parameters


def initialize_bnorm_params(parameters, bnorm_list, apply_momentum):
    L = parameters['numLayers']
    parameters['bnorm_list'] = bnorm_list
    for l in range(L):
        n = parameters["W" + str(l + 1)].shape[0]
        if bnorm_list[l]:
            parameters['bnorm_beta' + str(l + 1)] = np.zeros((n, 1))
            parameters['bnorm_gamma' + str(l + 1)] = np.ones((n, 1))
            if apply_momentum:
                parameters['Vbnorm_beta' + str(l + 1)] = np.zeros((n, 1))
                parameters['Gbnorm_beta' + str(l + 1)] = np.zeros((n, 1))
                parameters['Vbnorm_gamma' + str(l + 1)] = np.zeros((n, 1))
                parameters['Gbnorm_gamma' + str(l + 1)] = np.zeros((n, 1))
        else:
            parameters['bnorm_beta' + str(l + 1)] = np.asarray([])
            parameters['Vbnorm_beta' + str(l + 1)] = np.asarray([])
            parameters['Gbnorm_beta' + str(l + 1)] = np.asarray([])
            parameters['bnorm_gamma' + str(l + 1)] = np.asarray([])
            parameters['Vbnorm_gamma' + str(l + 1)] = np.asarray([])
            parameters['Gbnorm_gamma' + str(l + 1)] = np.asarray([])
    return parameters

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = {}
    cache["A"] = A_prev
    return Z, cache


def layer_forward(A_prev, W, b, activation, drop_prob, bnorm_beta, bnorm_gamma, mode):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    A, bnorm_cache = batchnorm(A, bnorm_beta,bnorm_gamma)
    A, drop_cache = dropout(A, drop_prob,mode)
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    cache["bnorm_cache"] = bnorm_cache
    cache["drop_cache"] = drop_cache
    return A, cache


def multi_layer_forward(A0, parameters, mode):
    L = parameters['numLayers']
    A = A0
    caches = []
    for l in range(L):
        A, cache = layer_forward(A, parameters["W" + str(l + 1)], parameters["b" + str(l + 1)],parameters["act" + str(l + 1)], parameters["dropout" + str(l + 1)],parameters['bnorm_beta' + str(l + 1)], parameters['bnorm_gamma' + str(l + 1)], mode)
        caches.append(cache)
    return A, caches

def linear_backward(dZ, cache, W, b):
    A = cache["A"]
    m = A.shape[1]
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    drop_cache = cache["drop_cache"]
    bnorm_cache = cache["bnorm_cache"]
    dA = dropout_der(dA, drop_cache)
    dA, dbnorm_beta, dbnorm_gamma = batchnorm_der(dA, cache["bnorm_cache"])
    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db, dbnorm_beta, dbnorm_gamma

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches)
    gradients = {}
    dA = dAL
    for l in reversed(range(L)):
        dA, gradients["dW"+str(l+1)], gradients["db"+str(l+1)],gradients["dbnorm_beta"+str(l+1)], gradients["dbnorm_gamma"+str(l+1)] =layer_backward(dA, caches[l], parameters["W"+str(l+1)],parameters["b"+str(l+1)], parameters["act"+str(l+1)])
    for key in gradients:
        gradients[key] = np.clip(gradients[key], -5, 5)
    return gradients


def update_parameters_with_momentum_Adam(parameters, gradients, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
    if 't' not in parameters:
        parameters['t'] = 1
    else:
        parameters['t'] += 1
    t = parameters['t']
    L = parameters['numLayers']
    apply_momentum = parameters['apply_momentum']
    bnorm_list = parameters['bnorm_list']

    for l in range(L):
        if apply_momentum:
            parameters["VdW" + str(l + 1)] = beta1 * parameters["VdW" + str(l + 1)] + (1 - beta1) * gradients["dW" + str(l + 1)]
            parameters["Vdb" + str(l + 1)] = beta1 * parameters["Vdb" + str(l + 1)] + (1 - beta1) * gradients["db" + str(l + 1)]
            parameters["GdW" + str(l + 1)] = beta2 * parameters["GdW" + str(l + 1)] + (1 - beta2) * (gradients["dW" + str(l + 1)] ** 2)
            parameters["Gdb" + str(l + 1)] = beta2 * parameters["Gdb" + str(l + 1)] + (1 - beta2) * (gradients["db" + str(l + 1)] ** 2)
            VdW_corr = parameters["VdW" + str(l + 1)] / (1 - beta1 ** t)
            Vdb_corr = parameters["Vdb" + str(l + 1)] / (1 - beta1 ** t)
            GdW_corr = parameters["GdW" + str(l + 1)] / (1 - beta2 ** t)
            Gdb_corr = parameters["Gdb" + str(l + 1)] / (1 - beta2 ** t)
            parameters["W" + str(l + 1)] -= alpha * VdW_corr / (np.sqrt(GdW_corr + eps))
            parameters["b" + str(l + 1)] -= alpha * Vdb_corr / (np.sqrt(Gdb_corr + eps))
        else:
            parameters["W" + str(l + 1)] -= alpha * gradients["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= alpha * gradients["db" + str(l + 1)]
        if apply_momentum and bnorm_list[l]:
            parameters['Vbnorm_beta' + str(l + 1)] = beta1 * parameters['Vbnorm_beta' + str(l + 1)] + (1 - beta1) * gradients["dbnorm_beta" + str(l + 1)]
            parameters['Vbnorm_gamma' + str(l + 1)] = beta1 * parameters['Vbnorm_gamma' + str(l + 1)] + (1 - beta1) * gradients["dbnorm_gamma" + str(l + 1)]
            parameters['Gbnorm_beta' + str(l + 1)] = beta2 * parameters['Gbnorm_beta' + str(l + 1)] + (1 - beta2) * ( gradients["dbnorm_beta" + str(l + 1)] ** 2)
            parameters['Gbnorm_gamma' + str(l + 1)] = beta2 * parameters['Gbnorm_gamma' + str(l + 1)] + (1 - beta2) * (gradients["dbnorm_gamma" + str(l + 1)] ** 2)
            Vbeta_corr = parameters['Vbnorm_beta' + str(l + 1)] / (1 - beta1 ** t)
            Vgamma_corr = parameters['Vbnorm_gamma' + str(l + 1)] / (1 - beta1 ** t)
            Gbeta_corr = parameters['Gbnorm_beta' + str(l + 1)] / (1 - beta2 ** t)
            Ggamma_corr = parameters['Gbnorm_gamma' + str(l + 1)] / (1 - beta2 ** t)
            parameters['bnorm_beta' + str(l + 1)] -= alpha * Vbeta_corr / (np.sqrt(Gbeta_corr + eps))
            parameters['bnorm_gamma' + str(l + 1)] -= alpha * Vgamma_corr / (np.sqrt(Ggamma_corr + eps))
            parameters['bnorm_gamma' + str(l + 1)] = np.clip(parameters['bnorm_gamma' + str(l + 1)], -10, 10)
        elif bnorm_list[l]:
            parameters['bnorm_beta' + str(l + 1)] -= alpha * gradients["dbnorm_beta" + str(l + 1)]
            parameters['bnorm_gamma' + str(l + 1)] -= alpha * gradients["dbnorm_gamma" + str(l + 1)]
            parameters['bnorm_gamma' + str(l + 1)] = np.clip(parameters['bnorm_gamma' + str(l + 1)], -10, 10)
    return parameters


def multi_layer_network(X, Y, net_dims, act_list, drop_prob_list, bnorm_list, num_epochs=3, batch_size=64,learning_rate=0.2, decay_rate=0.01, apply_momentum=True, log=True, log_step=200):
    mode = 'train'
    n, m = X.shape
    parameters = initialize_network(net_dims, act_list, drop_prob_list)
    parameters = initialize_velocity(parameters, apply_momentum)
    parameters = initialize_bnorm_params(parameters, bnorm_list, apply_momentum)
    parameters['apply_momentum'] = apply_momentum
    parameters['bnorm_list'] = bnorm_list
    costs = []
    itr = 1
    for epoch in range(num_epochs):
        alpha = learning_rate * (1 / (
                    1 + decay_rate * epoch))
        if log:
            print('------- Epoch {} -------'.format(epoch + 1))
        for ii in range((m - 1) // batch_size + 1):
            Xb = X[:, ii * batch_size: (ii + 1) * batch_size]
            Yb = Y[:, ii * batch_size: (ii + 1) * batch_size]
            A0 = Xb
            AL, caches = multi_layer_forward(A0, parameters, mode)
            AL, softmax_cache, cost = softmax_cross_entropy_loss(AL,Yb)
            dAL = softmax_cross_entropy_loss_der(Yb,softmax_cache)
            grads = multi_layer_backward(dAL, caches,parameters)
            parameters = update_parameters_with_momentum_Adam(parameters, grads,alpha)
            if itr % log_step == 0:
                costs.append(cost)
                if log:
                    print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (itr, cost, alpha))
            itr += 1
    return costs, parameters


def classify(X, parameters, mode='test'):
    AL, _ = multi_layer_forward(X, parameters,mode)
    AL_softmax, _, _ = softmax_cross_entropy_loss(AL, np.array([]))
    YPred = np.argmax(AL_softmax, axis=0, keepdims=True)
    return YPred

net_dims = [784, 100, 100, 64, 10]
drop_prob_list = [0.3, 0.3, 0, 0]
bnorm_list = [1,1,1,1]
assert(len(bnorm_list) == len(net_dims)-1)
act_list = ['relu', 'relu', 'relu', 'linear']
assert(len(act_list) == len(net_dims)-1)
num_epochs = 3
batch_size = 64
learning_rate = 1e-2
decay_rate = 0
apply_momentum = True
np.random.seed(1)
print("Network dimensions are:" + str(net_dims))
print('Dropout= [{}], Batch Size = {}, lr = {}, decay rate = {}'.format(drop_prob_list,batch_size,learning_rate,decay_rate))

trX, trY, tsX, tsY = samples_take()
costs, parameters = multi_layer_network(trX, trY, net_dims, act_list, drop_prob_list, bnorm_list, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,decay_rate=decay_rate, apply_momentum=apply_momentum, log=True)
train_Pred = classify(trX, parameters)
test_Pred = classify(tsX, parameters)
trAcc = np.mean(train_Pred == trY) * 100
teAcc = np.mean(test_Pred == tsY) * 100

print("Accuracy for training set is {0:0.3f} %".format(trAcc))
print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

plt.plot(range(len(costs)),costs)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()















