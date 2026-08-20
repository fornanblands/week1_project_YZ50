import matplotlib.pyplot as plt

def neuron_forward(x, w, b):
    z = 0
    for i in range(len(x)):
        z += x[i] * w[i]
    z += b
    return z

def sigmoid(z):
    e = 2.718281828459045
    return 1 / (1 + e ** (-z))

def layer_forward(x, w, b):
    outputs = []
    for i in range(len(w)):
        z = neuron_forward(x, w[i], b[i])
        y = sigmoid(z)
        outputs.append(y)
    return outputs

def mse_loss(p, t):
    total = 0
    for i in range(len(p)):
        total += (p[i] - t[i]) ** 2
    return total / len(p)

def plot_loss_curve():
    X = [0.5, -1.2, 3.0]
    T = [0.5, 0.0, 1.0]
    W = [[0.4, 0.7, -0.1], [-0.3, 0.2, 0.9], [0.1, -0.5, 0.6]]
    B = [0.1, -0.2, 0.05]
    Wv, Lv = [], []
    n = 100
    lo, hi = -5.0, 5.0
    for i in range(n + 1):
        wc = lo + (hi - lo) * i / n
        Wi = [[wc, W[0][1], W[0][2]], W[1], W[2]]
        p = layer_forward(X, Wi, B)
        L = mse_loss(p, T)
        Wv.append(wc)
        Lv.append(L)
    plt.figure()
    plt.plot(Wv, Lv)
    plt.xlabel("w[0][0]")
    plt.ylabel("loss")
    plt.title("Loss vs. w[0][0]")
    plt.grid(True)
    plt.show()

plot_loss_curve()
