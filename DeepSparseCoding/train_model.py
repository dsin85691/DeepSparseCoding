import os
import sys
import argparse
import time as ti

import torchvision

ROOT_DIR = os.path.dirname(os.getcwd())
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

import DeepSparseCoding.utils.loaders as loaders
import DeepSparseCoding.utils.run_utils as run_utils
import DeepSparseCoding.utils.dataset_utils as dataset_utils
import numpy as np
import torch
from DeepSparseCoding.models.ensemble_model import NetTensorFlowWrapper
from PIL import Image
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', help='Path to the parameter file')

args = parser.parse_args()
param_file = args.param_file

t0 = ti.time()

# Load params
params = loaders.load_params(param_file)

# Load data
train_loader, val_loader, test_loader, data_stats = dataset_utils.load_dataset(params)
for key, value in data_stats.items():
    print("KEY:", key)
    print("VALUE:", value)
    setattr(params, key, value)

# Load model
model = loaders.load_model(params.model_type, params)
model.to(params.device)
transposed_model = NetTensorFlowWrapper(model)

# rgb = torchvision.transforms.Grayscale(num_output_channels=3)
# torch_tensor = torchvision.transforms.ToTensor()
#
# train_dataset = None
# train_dataset_labels = None
#
# classes = ('0', '1', '2', '3', '4',
#            '5', '6', '7', '8', '9')
#
# for count, data in enumerate(train_loader):
#     images, labels = data
#
#     input_data = torch.transpose(images, 1, 3)
#
#     numpy_images = np.array(input_data)
#
#     for j in range(input_data.shape[0]):
#         numpyData = numpy_images[j]
#
#         im = Image.fromarray((numpyData[0] * 255).astype(np.uint8))
#
#         DataRGB = rgb.forward(im)
#
#         newData = torch_tensor(DataRGB)
#
#         numpy_data = newData.numpy()
#
#         numpy_data = np.reshape(numpy_data, (3, 784))
#
#         numpy_data = np.expand_dims(numpy_data, axis=0)
#
#         numpy_data = np.expand_dims(numpy_data, axis=0)
#
#         numpy_data = np.swapaxes(numpy_data, 1, 3)
#
#         numpy_data = np.swapaxes(numpy_data, 2, 3)
#
#         numpy_data = np.reshape(numpy_data, (1, 28, 28, 3))
#
#         if j == 0:
#             numpy_batch = np.array(numpy_data)
#         else:
#             numpy_batch = np.append(numpy_batch, numpy_data, axis=0)
#
#     if count == 0:
#         train_dataset = numpy_batch
#         train_dataset_labels = labels
#         train_dataset_labels = torch.reshape(train_dataset_labels, (1, labels.size()[0]))
#     else:
#         train_dataset = np.append(train_dataset, numpy_batch, axis=0)
#         labels = torch.reshape(labels, (1, labels.size()[0]))
#         train_dataset_labels = np.append(train_dataset_labels, labels, axis=0)
#
# train_dataset_labels = np.reshape(train_dataset_labels, (60000,1))
# print(train_dataset.shape)
# print(train_dataset_labels.shape)
#
# test_dataset = None
# test_dataset_labels = None
#
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
#
# for count, data in enumerate(test_loader):
#     images, labels = data
#
#     input_data = torch.transpose(images, 1, 3)
#
#     numpy_images = np.array(input_data)
#
#     numpy_batch = None
#
#     for j in range(input_data.shape[0]):
#         numpyData = numpy_images[j]
#
#         im = Image.fromarray((numpyData[0] * 255).astype(np.uint8))
#
#         DataRGB = rgb.forward(im)
#
#         newData = torch_tensor(DataRGB)
#
#         numpy_data = newData.numpy()
#
#         numpy_data = np.reshape(numpy_data, (3, 784))
#
#         numpy_data = np.expand_dims(numpy_data, axis=0)
#
#         numpy_data = np.expand_dims(numpy_data, axis=0)
#
#         numpy_data = np.swapaxes(numpy_data, 1, 3)
#
#         numpy_data = np.swapaxes(numpy_data, 2, 3)
#
#         numpy_data = np.reshape(numpy_data, (1, 28, 28, 3))
#
#         if j == 0:
#             numpy_batch = np.array(numpy_data)
#         else:
#             numpy_batch = np.append(numpy_batch, numpy_data, axis=0)
#
#     if count == 0:
#         test_dataset = numpy_batch
#         test_dataset_labels = labels
#         test_dataset_labels = torch.reshape(test_dataset_labels, (1, labels.size()[0]))
#     else:
#         test_dataset = np.append(test_dataset, numpy_batch, axis=0)
#         labels = torch.reshape(labels, (1, labels.size()[0]))
#         test_dataset_labels = np.append(test_dataset_labels, labels, axis=0)
#
#         # print(numpy_images.shape)
#     # for i in range(numpy_images.shape[0]):
#     #     single_image = numpy_images[i]
#     #     single_image = np.reshape(single_image, (1, 784))
#     #     print(single_image)
#     #     print(single_image.shape)
#     #     outputs = model(torch.from_numpy(single_image))
#     #     print(outputs)
#     #     print(outputs.shape)
#     #     _, predicted = torch.max(outputs, 1)
#     #     c = (predicted == labels).squeeze()
#     #     total += labels.size(0)
#     #
#     #     correct += (predicted == labels).sum().item()
#
# print(test_dataset.shape)
# print(test_dataset_labels.shape)
# test_dataset_labels = np.reshape(test_dataset_labels, (10000, 1))
# print(test_dataset_labels.shape)
#
# class TransformedDataSet(Dataset):
#     def __init__(self, dataset, dataset_labels):
#         transformed_dataset = dataset
#         transformed_labels = dataset_labels
#         self.len = transformed_dataset.shape[0] // 3
#         self.x_data = torch.from_numpy(transformed_dataset)
#         self.y_data = torch.from_numpy(transformed_labels)
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#     def __len__(self):
#         return self.len
#
#
# training_set = TransformedDataSet(train_dataset, train_dataset_labels)
# train_loader = DataLoader(dataset=training_set, batch_size=params.batch_size, shuffle=params.shuffle_data,
#                           num_workers=0, pin_memory=False)
#
# test_set = TransformedDataSet(test_dataset, test_dataset_labels)
# test_loader = DataLoader(dataset=test_set, batch_size=params.batch_size, shuffle=params.shuffle_data, num_workers=0,
#                          pin_memory=False)

# Train model
for epoch in range(1, transposed_model.params.num_epochs + 1):
    run_utils.train_epoch(epoch, transposed_model[0], train_loader)
    if (transposed_model[0].params.model_type.lower() in ['mlp', 'ensemble']):
        run_utils.test_epoch(epoch, transposed_model[0], test_loader)
    transposed_model[0].log_info(f'Completed epoch {epoch}/{transposed_model[0].params.num_epochs}')
    print(f'Completed epoch {epoch}/{transposed_model[0].params.num_epochs}')

# (inputs1, targets1, outputs1) = run_utils.get_inputs_and_outputs(1, transposed_model, train_loader,
#                                                                      num_batches=params.batch_size)
#
# (inputs2, targets2, outputs2) = run_utils.get_inputs_and_outputs(1, transposed_model, test_loader,
#                                                                      num_batches=params.batch_size)
# print(len(inputs1))
# print(len(targets1))
# print(len(outputs1))
# print(len(inputs2))
# print(len(targets2))
# print(len(outputs2))
# torch_tensor = torchvision.transforms.ToTensor()

# correct = 0
# total = 0
# for i in range(len(targets1)):
#     print(targets1[i].size())
#     print(outputs1[i].size())
#     for j in range(params.batch_size):
#         if targets1[i][j] == outputs1[i][j]:
#             correct += 1
#         total += 1
# print((100.) * (correct / total))
#
# correct = 0
# total = 0
# for i in range(len(targets2)):
#     print(targets2[i].size())
#     print(outputs2[i].size())
#     for j in range(params.batch_size):
#         if targets1[i][j] == outputs1[i][j]:
#             correct += 1
#         total += 1
# print((100.) * (correct / total))


t1 = ti.time()
tot_time = float(t1 - t0)
tot_images = transposed_model[0].params.num_epochs * len(train_loader.dataset)
out_str = f'Training on {tot_images} images is complete. Total time was {tot_time} seconds.\n'
transposed_model[0].log_info(out_str)
print('Training Complete\n')

for epoch in range(1):
    running_loss = 0.0

# print(next(iter(train_loader)))
# for data, target in iter(train_loader):
#     print(data, target)
#     break
# print(np.array(next(iter(train_loader))[0]))
# print(np.array_equal(np.array(next(iter(train_loader))[0]), np.array(np.zeros(np.array(next(iter(train_loader))[0]).shape))))
# print(model.forward(np.reshape(next(iter(train_loader))[0], (100, 784))))
# print(train_loader[0][1])

# outputs = model(torch.from_numpy(np.array(np.reshape(next(iter(train_loader))[0], (100, 784)))))
classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')

# rgb = torchvision.transforms.Grayscale(num_output_channels=3)
# torch_tensor = torchvision.transforms.ToTensor()
train_loader, val_loader, test_loader, data_stats = dataset_utils.load_dataset(params)

correct = 0
total = 0
for i, data in enumerate(test_loader, 0):
    images, labels = data

    print(images.shape)

    # input_data = torch.transpose(images, 1, 3)

    # print(input_data.shape)

    numpy_images = np.array(images)

    numpy_batch = None

    # for j in range(input_data.shape[0]):
    #     numpyData = numpy_images[j]
    #
    #     im = Image.fromarray((numpyData[0] * 255).astype(np.uint8))
    #
    #     DataRGB = rgb.forward(im)
    #
    #     newData = torch_tensor(DataRGB)
    #
    #     numpy_data = newData.numpy()
    #
    #     numpy_data = np.reshape(numpy_data, (3, 784))
    #
    #     numpy_data = np.expand_dims(numpy_data, axis=0)
    #
    #     numpy_data = np.expand_dims(numpy_data, axis=0)
    #
    #     numpy_data = np.swapaxes(numpy_data, 1, 3)
    #
    #     numpy_data = np.swapaxes(numpy_data, 2, 3)
    #
    #     numpy_data = np.reshape(numpy_data, (1, 28, 28, 3))
    #
    #     if j == 0:
    #         numpy_batch = np.array(numpy_data)
    #     else:
    #         numpy_batch = np.append(numpy_batch, numpy_data, axis=0)

        # print(numpy_images.shape)
    outputs = transposed_model(torch.from_numpy(numpy_images))

    predicted = outputs.max(1, keepdim=True)[1]
    labels = torch.reshape(labels, (100,))


    for i in range(len(labels)):
        if predicted[i][0] == labels[i]:
            correct += 1
        total += 1
    # for i in range(numpy_images.shape[0]):
    #     single_image = numpy_images[i]
    #     single_image = np.reshape(single_image, (1, 784))
    #     print(single_image)
    #     print(single_image.shape)
    #     outputs = model(torch.from_numpy(single_image))
    #     print(outputs)
    #     print(outputs.shape)
    #     _, predicted = torch.max(outputs, 1)
    #     c = (predicted == labels).squeeze()
    #     total += labels.size(0)
    #
    #     correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * (correct / total)))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data

        # print(images.shape)
        #
        # input_data = torch.transpose(images, 1, 3)
        #
        # print(input_data.shape)
        #
        # numpy_images = np.array(input_data)
        #
        # numpy_batch = None
        #
        # for j in range(input_data.shape[0]):
        #     numpyData = numpy_images[j]
        #
        #     im = Image.fromarray((numpyData[0] * 255).astype(np.uint8))
        #
        #     DataRGB = rgb.forward(im)
        #
        #     newData = torch_tensor(DataRGB)
        #
        #     numpy_data = newData.numpy()
        #
        #     numpy_data = np.reshape(numpy_data, (3, 784))
        #
        #     numpy_data = np.expand_dims(numpy_data, axis=0)
        #
        #     numpy_data = np.expand_dims(numpy_data, axis=0)
        #
        #     numpy_data = np.swapaxes(numpy_data, 1, 3)
        #
        #     numpy_data = np.swapaxes(numpy_data, 2, 3)
        #
        #     numpy_data = np.reshape(numpy_data, (1, 28, 28, 3))
        #
        #     if j == 0:
        #         numpy_batch = np.array(numpy_data)
        #     else:
        #         numpy_batch = np.append(numpy_batch, numpy_data, axis=0)

            # print(numpy_images.shape)
        outputs = transposed_model(images)

        predicted = outputs.max(1, keepdim=True)[1]
        predicted = torch.squeeze(predicted, 1)
        labels = torch.reshape(labels, (100,))

        # print(predicted)
        # print(labels)

        c = (predicted == labels).squeeze()

        print(c.size())

        for i in range(4):
            label = labels[i]

            class_correct[label] += c[i].item()
            class_total[label] += 1
            # for i in range(numpy_images.shape[0]):
            #     single_image = numpy_images[i]
            #     single_image = np.reshape(single_image, (1, 784))
            #     outputs = model(torch.from_numpy(single_image))
            #     _, predicted = torch.max(outputs, 1)
            #     c = (predicted == labels).squeeze()
            #     for i in range(4):
            #         label = labels
            #         class_correct[label] += c.item()
            #         class_total[label] += 1
            # print(numpy_images.shape)
            # numpy_images = np.reshape(numpy_images, (50, 784))
            # outputs = model(torch.from_numpy(numpy_images))
            # _, predicted = torch.max(outputs, 1)
            # c = (predicted == labels).squeeze()
            # for i in range(4):
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

transposed_model[0].write_checkpoint()
