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

x = [0.5, -1.2, 3.0]
w = [[0.4, 0.7, -0.1], [-0.3, 0.2, 0.9], [0.1, -0.5, 0.6]]
b = [0.1, -0.2, 0.05]
t = [0.5, 0.0, 1.0]
p = layer_forward(x, w, b)
print("p =", p)
loss = mse_loss(p, t)
print("loss =", loss)
