import torchvision.datasets

import transform_c

train_data = torchvision.datasets.StanfordCars(root='./', split='train',
                                               transform=transform_c.train_transform,
                                               download=True)
test_data = torchvision.datasets.StanfordCars(root='./', split='test',
                                              transform=transform_c.test_transform,
                                              download=True)
dataset = {'train': train_data, 'test': test_data}
if __name__ == '__main__':
    print('train len:', len(train_data))
    print('test len:', len(test_data))
