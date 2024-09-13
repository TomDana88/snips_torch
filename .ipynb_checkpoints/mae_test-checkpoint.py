from datasets.celeba import CelebA

dataset = CelebA(root=os.path.join(args.exp, 'datasets'), split='test',
                 transform=transforms.Compose([transforms.CenterCrop(140),
                     transforms.Resize(config.data.image_size),
                     transforms.ToTensor(),
                 ]), download=False)
dataloader = Dataloader(
    dataset,
    bath_size=1,
    shuffle=True,
    num_workers=0,
)

samples = next(iter(dataloader))
print(samples.shape)
