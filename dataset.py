import torchvision

# Скачиваем датасет EMNIST, split='letters'
# torchvision сам проверит MD5 архива при загрузке
dataset = torchvision.datasets.EMNIST(
    root='./data',          # папка, куда сохранится датасет
    split='letters',        # именно Letters
    train=True,             # скачает и train, и test
    download=True
)
