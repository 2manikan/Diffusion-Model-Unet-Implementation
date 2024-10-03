# Diffusion-Model-Unet-Implementation

A modified U-Net architecture implementation written from scratch with Residual Blocks as layers and timestep conditioning. It is a simple use-case of the Hugging Face Diffusion Model U-Net (UNet2DModel) and provides the following additional benefits:
        (1) Skip Connection Architecture Customization/Modification
        (2) Customization of the Middle Block Structure
        (3) Elimination of additional optional convolution layers if preferred.

The same training procedure from the IADB(Iterative a-Deblending) model was used to train the U-Net. The model was used on rip current data from the Webcam Coastal Observation System to generate cloudy ocean images to create a dataset for future deep learning. The generation sequence is included.

Credits:
https://github.com/huggingface/diffusers/tree/main,
https://github.com/tchambon/IADB, and
https://webcoos.org/
