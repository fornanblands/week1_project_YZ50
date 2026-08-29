import math


class value: 

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f'value(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad 
        out._backward = _backward

        return out

    def __radd__(self, other): 
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self. data * other.data, (self,other), '*')

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward

        return out

    def __rmul__(self,other):
        return self * other


    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = value(self.data ** power, (self,),  '**')


        def _backward():
            self.grad += (power * self.data ** (power -1)) * out.grad
        out._backward = _backward

        return out


    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1
 
    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        out = value(math.exp(self.data), (self,), 'exp')
 
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
 
        return out
    

    def backward(self):
        topo = []
        visited = set()
 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
 
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
import torch


X1, X2, W1, W2, B = 2.0, 0.0, -3.0, 1.0, 6.8813735870195432
h = 1e-6

x1, x2 = value(X1), value(X2)
w1, w2 = value(W1), value(W2)
b = value(B)

n = x1*w1 + x2*w2 + b
e = (2 * n).exp()
( (e - 1) / (e + 1) ).backward()



x1_pt = torch.tensor([X1], dtype=torch.float64, requires_grad=True)
x2_pt = torch.tensor([X2], dtype=torch.float64, requires_grad=True)
w1_pt = torch.tensor([W1], dtype=torch.float64, requires_grad=True)
w2_pt = torch.tensor([W2], dtype=torch.float64, requires_grad=True)
b_pt  = torch.tensor([B], dtype=torch.float64, requires_grad=True)

torch.tanh(x1_pt*w1_pt + x2_pt*w2_pt + b_pt).backward()



def f(x1, x2, w1, w2, b):
    return math.tanh(x1*w1 + x2*w2 + b)

num_x1 = (f(X1+h, X2, W1, W2, B) - f(X1-h, X2, W1, W2, B)) / (2*h)
num_w1 = (f(X1, X2, W1+h, W2, B) - f(X1, X2, W1-h, W2, B)) / (2*h)
num_x2 = (f(X1, X2+h, W1, W2, B) - f(X1, X2-h, W1, W2, B)) / (2*h)
num_w2 = (f(X1, X2, W1, W2+h, B) - f(X1, X2, W1, W2-h, B)) / (2*h)
num_b  = (f(X1, X2, W1, W2, B+h) - f(X1, X2, W1, W2, B-h)) / (2*h)



print(f"{'Değişken':<10} | {'tanh parcalanmis':<15} | {'PyTorch':<15} | {'derivative context':<15}")
print("-" * 62)
print(f"{'x1.grad':<10} | {x1.grad:<15.6f} | {x1_pt.grad.item():<15.6f} | {num_x1:<15.6f}")
print(f"{'w1.grad':<10} | {w1.grad:<15.6f} | {w1_pt.grad.item():<15.6f} | {num_w1:<15.6f}")
print(f"{'x2.grad':<10} | {x2.grad:<15.6f} | {x2_pt.grad.item():<15.6f} | {num_x2:<15.6f}")
print(f"{'w2.grad':<10} | {w2.grad:<15.6f} | {w2_pt.grad.item():<15.6f} | {num_w2:<15.6f}")
print(f"{'b.grad':<10} | {b.grad:<15.6f} | {b_pt.grad.item():<15.6f} | {num_b:<15.6f}")
