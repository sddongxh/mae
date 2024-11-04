import math
import matplotlib.pyplot as plt

def adjust_learning_rate_iteration(optimizer, iteration, warmup_iters, total_iters, min_lr, max_lr):
    """Decay the learning rate with half-cycle cosine after warmup, based on iteration count"""
    if iteration < warmup_iters:
        lr = max_lr * iteration / warmup_iters
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (iteration - warmup_iters) / (total_iters - warmup_iters)))
    
    # Apply the learning rate to each parameter group, considering possible lr_scale
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# 示例参数和优化器结构
class DummyOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.1, "lr_scale": 1.0}, {"lr": 0.1}]

# Example usage
optimizer = DummyOptimizer()  # Assuming a defined optimizer with param_groups
warmup_iters = 500
total_iters = 5000
min_lr = 0.001
max_lr = 0.1

# Simulate the learning rate adjustment over iterations
lr_values_iter = [adjust_learning_rate_iteration(optimizer, iteration, warmup_iters, total_iters, min_lr, max_lr)
                  for iteration in range(total_iters)]




# # 定义参数
# class IterArgs:
#     def __init__(self):
#         self.lr = 0.1           # 初始学习率
#         self.min_lr = 0.001     # 最小学习率
#         self.warmup_iters = 500 # 预热的迭代次数
#         self.total_iters = 5000 # 总迭代次数

# args = IterArgs()
# optimizer = DummyOptimizer()

# # 测试每次迭代的学习率调整
# lr_values_iter = [adjust_learning_rate_iteration(optimizer, iteration, args) for iteration in range(total_iters)]

# 绘制学习率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(total_iters), lr_values_iter, marker='o', markersize=2, linestyle='-')
plt.title("Learning Rate Schedule with Warmup and Cosine Decay (Iteration-based)")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()
