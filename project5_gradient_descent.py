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

X = [0.5, -1.2, 3.0]
T = [0.5, 0.0, 1.0]
W = [[0.4, 0.7, -0.1], [-0.3, 0.2, 0.9], [0.1, -0.5, 0.6]]
B = [0.1, -0.2, 0.05]

def L_w0(w0):
    Wi = [[w0, W[0][1], W[0][2]], W[1], W[2]]
    p = layer_forward(X, Wi, B)
    return mse_loss(p, T)

def grad_w0(w0, h=0.0001):
    return (L_w0(w0 + h) - L_w0(w0 - h)) / (2 * h)

w0 = 0.0
lr = 2.0
epochs = 25
for step in range(epochs):
    g = grad_w0(w0)
    w0 = w0 - lr * g
    L = L_w0(w0)
    print(f"step {step}: w0={round(w0,4)} loss={round(L,4)}")
