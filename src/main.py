from data import *
from clf import *
from vae import *
from torch.utils.data import DataLoader
import torchvision.transforms as T

def main() -> None:

    train_transforms = T.Compose([T.ToTensor(),
                                  T.RandomRotation(30),
                                  T.RandomVerticalFlip(),
                                  T.RandomHorizontalFlip(),
                                  T.GaussianBlur(3),
                                  T.ColorJitter()])
    test_transforms = T.Compose([T.ToTensor()])

    train_dataset = DataReader(TRAIN, train_transforms, load=0.1)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = DataReader(TEST, test_transforms, load=0.5)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    vae = VAE(256)

    train_vae(100, vae, train_loader, test_loader)

    torch.save(vae, "/models/best_vae.pth")

if __name__ == "__main__":
    main()