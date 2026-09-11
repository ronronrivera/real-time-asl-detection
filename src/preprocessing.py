from torchvision import transforms


MEAN = [0.2189, 0.1511, 0.1363]
STD = [0.2999, 0.2186, 0.2071]

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ]
)
