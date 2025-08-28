# tiny_numpy_cnn.py
import numpy as np

# --------- helpers ----------
def one_hot(y, num_classes):
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out

def softmax_logits(logits):
    # logits: (B, C)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    # mean CE over batch
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

# --------- layers ----------
class Conv2D:
    """
    Naive 2D conv (stride=1, padding='same'), NCHW or NHWC?
    We'll use NHWC: (B, H, W, C)
    W: (KH, KW, Cin, Cout), b: (Cout,)
    """
    def __init__(self, Cin, Cout, k=3, seed=0):
        rng = np.random.default_rng(seed)
        lim = np.sqrt(6/(Cin*k*k + Cout*k*k))
        self.W = rng.uniform(-lim, lim, size=(k, k, Cin, Cout)).astype(np.float32)
        self.b = np.zeros((Cout,), dtype=np.float32)
        self.k = k
        self.cache = None

    def forward(self, x):
        B, H, W, Cin = x.shape
        pad = self.k // 2
        xpad = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
        H2, W2 = H, W
        k = self.k
        Cout = self.W.shape[-1]
        y = np.zeros((B, H2, W2, Cout), dtype=np.float32)
        # very naive loops; fine for tiny demos
        for i in range(H2):
            for j in range(W2):
                patch = xpad[:, i:i+k, j:j+k, :]           # (B, k, k, Cin)
                # (B, k, k, Cin) * (k, k, Cin, Cout) -> (B, Cout)
                y[:, i, j, :] = (patch.reshape(B, -1) @ self.W.reshape(-1, Cout)) + self.b
        self.cache = (x, xpad)
        return y

    def backward(self, dy, lr):
        x, xpad = self.cache
        B, H, W, Cin = x.shape
        k = self.k
        Cout = self.W.shape[-1]
        pad = k // 2

        dW = np.zeros_like(self.W)
        db = np.sum(dy, axis=(0,1,2))  # (Cout,)
        dxpad = np.zeros_like(xpad)

        for i in range(H):
            for j in range(W):
                patch = xpad[:, i:i+k, j:j+k, :]                      # (B,k,k,Cin)
                # accumulate dW
                # (k*k*Cin, B) @ (B, Cout) -> (k*k*Cin, Cout)
                dW += (patch.reshape(B, -1).T @ dy[:, i, j, :]).reshape(k, k, Cin, Cout)
                # dxpad gets contribution from dy * W
                # (B, Cout) @ (Cout, k*k*Cin) -> (B, k*k*Cin)
                dxpatch = dy[:, i, j, :] @ self.W.reshape(-1, Cout).T
                dxpad[:, i:i+k, j:j+k, :] += dxpatch.reshape(B, k, k, Cin)

        # unpad
        dx = dxpad[:, pad:-pad, pad:-pad, :]
        # SGD update
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class ReLU:
    def __init__(self): self.mask = None
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, dy, lr):
        return dy * self.mask

class MaxPool2x2:
    def __init__(self):
        self.cache = None  # (mask, input_shape)

    def forward(self, x):
        # x: (B, H, W, C); H and W must be even
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0
        x_ = x.reshape(B, H//2, 2, W//2, 2, C)      # (B, H2, 2, W2, 2, C)

        # max over the 2x2 window
        y = x_.max(axis=(2, 4))                     # (B, H2, W2, C)

        # build a UNIQUE mask via argmax (break ties deterministically)
        flat = x_.reshape(B, H//2, W//2, 4, C)      # flatten 2x2 -> 4
        arg = flat.argmax(axis=3)                   # (B, H2, W2, C) in {0,1,2,3}

        mask = np.zeros_like(x_, dtype=bool)        # (B, H2, 2, W2, 2, C)
        mask[:, :, 0, :, 0, :] = (arg == 0)
        mask[:, :, 0, :, 1, :] = (arg == 1)
        mask[:, :, 1, :, 0, :] = (arg == 2)
        mask[:, :, 1, :, 1, :] = (arg == 3)

        self.cache = (mask, x.shape)
        return y

    def backward(self, dy, lr):
        # dy: (B, H2, W2, C)
        mask, in_shape = self.cache
        B, H, W, C = in_shape
        H2, W2 = H//2, W//2

        dx_ = np.zeros_like(mask, dtype=dy.dtype)   # (B, H2, 2, W2, 2, C)
        # scatter gradient to the winner position in each 2x2 window
        dx_[:, :, 0, :, 0, :] = dy * mask[:, :, 0, :, 0, :]
        dx_[:, :, 0, :, 1, :] = dy * mask[:, :, 0, :, 1, :]
        dx_[:, :, 1, :, 0, :] = dy * mask[:, :, 1, :, 0, :]
        dx_[:, :, 1, :, 1, :] = dy * mask[:, :, 1, :, 1, :]

        return dx_.reshape(B, H, W, C)

class Linear:
    def __init__(self, Din, Dout, seed=0):
        rng = np.random.default_rng(seed)
        lim = np.sqrt(6/(Din + Dout))
        self.W = rng.uniform(-lim, lim, size=(Din, Dout)).astype(np.float32)
        self.b = np.zeros((Dout,), dtype=np.float32)
        self.cache = None
    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b
    def backward(self, dy, lr):
        x = self.cache
        dW = x.T @ dy
        db = dy.sum(axis=0)
        dx = dy @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

# --------- model ----------
class TinyCNN:
    def __init__(self, num_classes=10, seed=0):
        self.conv = Conv2D(Cin=1, Cout=8, k=3, seed=seed)
        self.relu = ReLU()
        self.pool = MaxPool2x2()
        self.fc = Linear(Din=14*14*8, Dout=num_classes, seed=seed+1)

    def forward(self, x):                 # x: (B, 28, 28, 1)
        z = self.conv.forward(x)          # (B, 28,28,8)
        z = self.relu.forward(z)
        z = self.pool.forward(z)          # (B, 14,14,8)
        z = z.reshape(z.shape[0], -1)     # (B, 1568)
        logits = self.fc.forward(z)       # (B, 10)
        return logits

    def backward(self, dlogits, lr):
        dz = self.fc.backward(dlogits, lr)                     # (B, 1568)
        dz = dz.reshape(-1, 14, 14, 8)
        dz = self.pool.backward(dz, lr)                        # (B, 28,28,8)
        dz = self.relu.backward(dz, lr)
        dx = self.conv.backward(dz, lr)                        # (B, 28,28,1)
        return dx

# --------- training demo ----------
if __name__ == "__main__":
    # For quick demo we’ll pull MNIST via Keras loader (only for data)
    from tensorflow.keras.datasets import mnist
    (Xtr, ytr), (Xte, yte) = mnist.load_data()
    # normalize & shape to NHWC with 1 channel
    Xtr = (Xtr.astype(np.float32)/255.0)[:, :, :, None]
    Xte = (Xte.astype(np.float32)/255.0)[:, :, :, None]
    ytr_oh = one_hot(ytr, 10)
    yte_oh = one_hot(yte, 10)

    net = TinyCNN(num_classes=10)
    lr = 1e-3
    batch = 128
    epochs = 4

    for ep in range(epochs):
        # shuffle
        idx = np.random.permutation(len(Xtr))
        Xtr, ytr, ytr_oh = Xtr[idx], ytr[idx], ytr_oh[idx]
        # mini-batches
        for i in range(0, len(Xtr), batch):
            xb = Xtr[i:i+batch]
            yb = ytr[i:i+batch]
            yb_oh = ytr_oh[i:i+batch]

            logits = net.forward(xb)                # (B,10)
            probs = softmax_logits(logits)
            loss = cross_entropy_loss(probs, yb_oh)

            # gradient of CE-softmax w.r.t logits: (probs - y)/B
            dlogits = (probs - yb_oh) / xb.shape[0]
            net.backward(dlogits, lr)

        # quick val
        logits = net.forward(Xte[:2000])
        pred = softmax_logits(logits).argmax(1)
        acc = (pred == yte[:2000]).mean()
        print(f"epoch {ep+1}: val_acc≈{acc:.3f}, last_train_loss≈{loss:.3f}")

