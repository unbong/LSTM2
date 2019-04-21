from LSTM import TimeEmbedding
import numpy as np

weight = np.random.randn(100, 2)
xs = np.arange(100).reshape(4, 25)

# print(weight[0:2])
model = TimeEmbedding(weight)
result = model.forward(xs)
# print(result)
dout = np.random.randn(4, 25, 2)
print(dout[:, 0:5, :])

print("slice")
model.backward(dout)
print(model.grads[0:20])
