Learning Robot Locomotion from Diverse Datasets
## Getting Started
### Create Conda Environment
To create a Conda environment, follow these steps:
1. Install the dependencies listed in `requirements.txt`.
2. Use the following command to create a Conda environment, replacing `<env>` with your preferred name:
    ```bash
    conda create --name <env> --file requirements.txt
    ```
3. Activate the environment:
    ```bash
    conda activate <env>
    ```
Paper accepted at: The 12th International Symposium on Adaptive Motion of Animals and Machines and 2nd LokoAssist Symposium.
More details: https://www.ias.informatik.tu-darmstadt.de/uploads/Site/EditPublication/lu_liu_master_thesis.pdf

## Overview

This learning framework is built around three main components:

    Motion Retargeting

    Motion Representation

    Motion Learning

You'll find more details and code in each corresponding folder.

TL;DR of the pipeline:

    We begin by collecting several motion datasets from online sources and retargeting them to our target quadruped platforms (Unitree A1 and Go2). These retargeted sequences serve as the foundation for training a Vector Quantized Variational Autoencoder (VQ-VAE), which learns to reconstruct reference motions and forms a discrete latent space conducive to generative modeling.

    To generate motion sequences, we employ a GPT-based autoregressive model, inspired by sequence modeling in natural language processing. Rather than predicting continuous vectors, the model operates on discrete latent indices from the VQ-VAE codebook. This allows us to generate motion trajectories that are coherent and stylistically aligned with the original dataset, conditioned on high-level prompts.

    For control, we adopt an imitation learning approach based on DeepMimic, training a policy to follow the generated reference motions. The resulting policy demonstrates the ability to imitate 15 distinct gaits, each selected via a discrete command at the beginning of an episode.