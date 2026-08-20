def neuron_forward(x, w, b):
    z = 0
    for i in range(len(x)):
        z += x[i] * w[i]
    z += b
    return z

def sigmoid(z):
    e = 2.718281828459045
    return 1 / (1 + e ** (-z))

x = [0.5, -1.2, 3.0]
w = [0.4, 0.7, -0.1]
b = 0.1
z = neuron_forward(x, w, b)
print("z =", z)
y = sigmoid(z)
print("y =", y)
