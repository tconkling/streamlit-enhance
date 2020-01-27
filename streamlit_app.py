import streamlit as st

from training import TrainingOptions, DeviceOptions, load_model, open_image, super_resolve, resize_naive

"""
# Streamlit: ENHANCE

PyTorch [Super Resolution Example](https://github.com/pytorch/examples/tree/master/super_resolution) 
in Streamlit
"""

st.sidebar.markdown("## Training Options")

training_opts = TrainingOptions(
    upscale_factor=st.sidebar.slider('Upscale Factor', value=4, min_value=1, max_value=5),
    nEpochs=st.sidebar.slider('Training Epochs', value=35, min_value=1, max_value=100),
    lr=0.001,
    seed=123,
)

st.sidebar.markdown("## Hardware Options")

device_opts = DeviceOptions(
    cuda=st.sidebar.checkbox('Use CUDA', value=False),
    threads=st.sidebar.slider('Dataloader Threads', value=4, min_value=1, max_value=16),
    batchSize=st.sidebar.slider('Training Batch Size', value=4, min_value=1, max_value=256),
    testBatchSize=st.sidebar.slider('Testing Batch Size', value=100, min_value=1, max_value=256),
)

st.code(f"Using: upscale={training_opts.upscale_factor}x, "
        f"nEpochs={training_opts.nEpochs}")

model = load_model(training_opts, device_opts)

input_image = st.file_uploader("Upload an image", ["png", "jpg"], encoding=None)
if input_image is None:
    input_image = 'leon.png'

st.image(open_image(input_image))

st.write('Super Resolution:')
st.image(super_resolve(model, input_image, False))

st.write('Naive Upscale:')
st.image(resize_naive(input_image, training_opts.upscale_factor))
