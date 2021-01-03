from torchvision import transforms

from utils.opt import Options
from utils.dataset import get_datasets, get_dataloader


opt = Options().parse()

transform = transforms.Compose([
    transforms.ToTensor()
])

datasets = get_datasets(opt.data_dir, transform=transform)
data = get_dataloader(data=datasets, batch_size=4, shuffle=True, num_workers=4)


