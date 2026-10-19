class GradientDescent:
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]
