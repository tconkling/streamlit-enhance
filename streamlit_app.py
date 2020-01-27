from collections import namedtuple
from io import BytesIO
from math import log10
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import PIL.Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from super_resolution.data import get_training_set, get_test_set
from super_resolution.model import Net

Opt = namedtuple('Opt', [
    'upscale_factor',
    'batchSize',
    'testBatchSize',
    'nEpochs',
    'lr',
    'cuda',
    'threads',
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


@st.cache(persist=True, suppress_st_warning=True)
def train(opt: Opt) -> bytes:
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_set(opt.upscale_factor)
    test_set = get_test_set(opt.upscale_factor)
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscale_factor).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

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

            training_status.code("===> Epoch[{}/{}]({}/{}): Loss: {:.4f}"
                                 .format(epoch, opt.nEpochs, iteration, len(training_data_loader), loss.item()))

        training_status.code("===> Epoch {} Complete: Avg. Loss: {:.4f}"
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

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        # test()

    bytes = BytesIO()
    torch.save(model, bytes)
    bytes.seek(0)
    bytestring = bytes.getbuffer().tobytes()
    bytes.close()
    return bytestring


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


"""
# Streamlit: ENHANCE

PyTorch [Super Resolution Example](https://github.com/pytorch/examples/tree/master/super_resolution) 
in Streamlit
"""

st.sidebar.markdown("## Training Params")

opt = Opt(
    cuda=st.sidebar.checkbox('Use CUDA', value=False),
    upscale_factor=st.sidebar.slider('Upscale Factor', value=4, min_value=1, max_value=5),
    batchSize=st.sidebar.slider('Training Batch Size', value=4, min_value=1, max_value=256),
    testBatchSize=st.sidebar.slider('Testing Batch Size', value=100, min_value=1, max_value=256),
    nEpochs=st.sidebar.slider('Training Epochs', value=35, min_value=1, max_value=100),
    lr=0.001,
    threads=st.sidebar.slider('Dataloader Threads', value=4, min_value=1, max_value=16),
    seed=123,
)

model_bytes = train(opt)
model = torch.load(BytesIO(model_bytes))

# Lena: https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/220px-Lenna_%28test_image%29.png
# Car: dataset/BSDS300/images/test/21077.jpg
# Corn: dataset/BSDS300/images/test/58060.jpg

st.code(f"Using: upscale={opt.upscale_factor}x, nEpochs={opt.nEpochs}")

input_image = st.file_uploader("Upload an image", ["png", "jpg"], encoding=None)
if input_image is None:
    input_image = 'leon.png'

st.image(open_image(input_image))

st.write('Super Resolution:')
st.image(super_resolve(model, input_image, False))

st.write('Naive Upscale:')
st.image(resize_naive(input_image, opt.upscale_factor))
