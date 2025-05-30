# Comparación rápida
import torch, time

x = torch.rand(100000, device='cuda')
y = torch.rand(100000, device='cuda')

start = time.time()
for _ in range(1000):
    z = (x * y).sum()
    val = z.item()  # <-- cada .item() fuerza espera GPU → CPU
end = time.time()
print("Con .item():", end - start)

start = time.time()
for _ in range(1000):
    z = (x * y).sum()
    val = z.detach()  # <-- no sincroniza
end = time.time()
print("Con .detach():", end - start)
