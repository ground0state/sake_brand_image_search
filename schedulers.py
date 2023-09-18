# %%
import torch
import math
from matplotlib import pyplot as plt


class CustomCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs, constant_epochs, cosine_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.constant_epochs = constant_epochs
        self.cosine_epochs = cosine_epochs
        self.eta_min = eta_min
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_epochs + self.constant_epochs:
            # Constant lr phase
            return [base_lr for base_lr in self.base_lrs]

        else:
            # Cosine annealing phase
            epochs_elapsed = self.last_epoch - self.warmup_epochs - self.constant_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * epochs_elapsed / self.cosine_epochs)) / 2
                    for base_lr in self.base_lrs]


# 例
num_epochs = 50
constant_epochs = 10
optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
scheduler = CustomCosineAnnealingLR(
    optimizer,
    warmup_epochs=0,
    constant_epochs=constant_epochs,
    cosine_epochs=num_epochs-constant_epochs,
    eta_min=0.001)

lrs = []
for epoch in range(50):
    lr_ = scheduler.get_last_lr()[0]
    lrs.append(lr_)
    print(epoch, lr_)
    optimizer.step()
    scheduler.step()

# %%
plt.plot(lrs)
# %%
