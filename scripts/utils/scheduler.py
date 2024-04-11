

class RLScheduler:
    def __init__(self):
        pass

    def update(self,current_lr, epoch, kl_dist):
        pass

class IdentityScheduler(RLScheduler):
    def __init__(self):
        super().__init__()

     
    def update(self, current_lr, epoch, kl_dist):
        return current_lr  


class AdaptiveScheduler(RLScheduler):
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, epoch, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr         


class LinearScheduler(RLScheduler):
    def __init__(self, start_lr, min_lr=1e-6, max_epochs = 1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs

    def update(self, current_lr, epoch, kl_dist):
        mul = epoch / self.max_epochs
        lr = max(self.min_lr, self.start_lr + (self.min_lr - self.start_lr) * mul)
        return lr