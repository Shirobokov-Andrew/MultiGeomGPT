from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

# 1. Создаем writer (логи будут сохраняться в ./runs/)
writer = SummaryWriter("tensorboard_logs/experiment_2")

# 2. Логируем метрики (например, в цикле обучения)
for epoch in tqdm(range(1, 1001)):
    loss = 1 / epoch  # ваша функция потерь
    accuracy = epoch  # метрика точности
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)
    time.sleep(2)

# 3. Закрываем writer
writer.close()