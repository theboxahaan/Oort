# -*- coding: utf-8 -*-

import sys

from torchvision import transforms


def get_data_transform(data: str):
    if data == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif data == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # input arguments: length&width of a figure
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),   # input arguments: length&width of a figure
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'emnist':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            #transforms.Resize(224),   # input arguments: length&width of a figure
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            #transforms.Resize(224),
            #transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif data == 'openImg':
        train_transform = transforms.Compose([
            transforms.Resize((64,64)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Resize(224),   # input arguments: length&width of a figure
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((64,64)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        print('Data must be {} or {} !'.format('mnist', 'cifar'))
        sys.exit(-1)

    return train_transform, test_transform
