from collections import namedtuple
from io import BytesIO
from math import log10
from typing import Any
from urllib.parse import urlparse

import PIL.Image
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from app_cache import AppCache
from super_resolution.data import get_training_set, get_test_set
from super_resolution.model import Net

model_cache = AppCache("trained_models")

DeviceOptions = namedtuple('DeviceOptions', [
    'cuda',
    'threads',
    'batchSize',
    'testBatchSize',
])

TrainingOptions = namedtuple('TrainingOptions', [
    'upscale_factor',
    'nEpochs',
    'lr',
    'seed',
])


def open_image(src: Any) -> PIL.Image.Image:
    if isinstance(src, PIL.Image.Image):
        return src

    if isinstance(src, str):
        try:
            p = urlparse(src)
            if p.scheme:
                response = requests.get(src)
                return PIL.Image.open(BytesIO(response.content))
        except UnicodeDecodeError:
            pass

    try:
        return PIL.Image.open(src)
    except:
        pass

    raise RuntimeError('Unrecognized image: %s' % src)


def load_model(training_opt: TrainingOptions, device_opt: DeviceOptions) -> Net:
    """Load a trained model from cache. If the model doesn't exist yet,
    train it and cache it."""
    model_bytes = model_cache.read(training_opt)
    if model_bytes is not None:
        return torch.load(BytesIO(model_bytes))

    model = train(training_opt, device_opt)
    bytes = BytesIO()
    torch.save(model, bytes)
    model_cache.write(training_opt, bytes.getvalue())
    bytes.close()
    return model


def train(training_opt: TrainingOptions, device_opt: DeviceOptions) -> Net:
    """Train a new model using the given options."""
    if device_opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(training_opt.seed)

    device = torch.device("cuda" if device_opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_set(training_opt.upscale_factor)
    test_set = get_test_set(training_opt.upscale_factor)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=device_opt.threads,
        batch_size=device_opt.batchSize,
        shuffle=True)
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=device_opt.threads,
        batch_size=device_opt.testBatchSize,
        shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=training_opt.upscale_factor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=training_opt.lr)

    training_status = st.empty()
    test_status = st.empty()

    def train(epoch: int):
        training_status.code('Training... (epoch=%s)' % epoch)
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            loss = criterion(model(input), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            training_status.code(
                "===> Epoch[{}/{}]({}/{}): Loss: {:.4f}"
                    .format(epoch, training_opt.nEpochs, iteration, len(training_data_loader), loss.item()))

        training_status.code(
            "===> Epoch {} Complete: Avg. Loss: {:.4f}"
                .format(epoch, epoch_loss / len(training_data_loader)))

    def test():
        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input, target = batch[0].to(device), batch[1].to(device)

                prediction = model(input)
                mse = criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        test_status.code("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    for epoch in range(1, training_opt.nEpochs + 1):
        train(epoch)
        # test()

    return model


def resize_naive(input_image: Any, scale_factor: int) -> PIL.Image.Image:
    img = open_image(input_image)
    return img.resize((img.size[0] * scale_factor, img.size[1] * scale_factor), PIL.Image.ANTIALIAS)


def super_resolve(model: Net, input_image: Any, cuda: bool) -> PIL.Image.Image:
    img = open_image(input_image).convert('YCbCr')
    y, cb, cr = img.split()

    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if cuda:
        model = model.cuda()
        input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img