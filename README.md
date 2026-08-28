# EX. NO. 6

# IMPLEMENTING A GAN

## AIM

To understand and implement a Generative Adversarial Network (GAN) using PyTorch and train it on the CIFAR-10 dataset to generate realistic images.

---

# PAGE 1 – INTRODUCTION

## INTRODUCTION

A **Generative Adversarial Network (GAN)** is a type of deep learning model used to generate new data that resembles the original training data.

A GAN consists of two neural networks:

### 1. Generator

The Generator creates fake images from a random noise vector. Its objective is to generate images that look realistic enough to fool the Discriminator.

### 2. Discriminator

The Discriminator acts as a binary classifier. It receives both real images from the dataset and fake images produced by the Generator and determines whether an image is real or fake.

The Generator and Discriminator compete with each other during training. As training progresses, the Generator improves its ability to produce realistic images.

## DATASET

The **CIFAR-10 dataset** is used in this experiment.

CIFAR-10 contains:

* 50,000 training images
* 10,000 test images
* 10 object classes
* Image size: 32 × 32 pixels
* Colour images with 3 RGB channels

## GAN WORKING

```text
Random Noise
     ↓
 Generator
     ↓
Fake Image ──────┐
                 ↓
             Discriminator
                 ↑
Real Image ──────┘
                 ↓
           Real / Fake
```

The Generator tries to fool the Discriminator, while the Discriminator tries to correctly identify real and generated images.

---

# PAGE 2 – STEP 1: IMPORTING REQUIRED LIBRARIES

## STEP 1: IMPORTING REQUIRED LIBRARIES

The required PyTorch, torchvision, NumPy, and Matplotlib libraries are imported.

### PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```

## DESCRIPTION

* **torch** – Provides the core PyTorch functionality.
* **torch.nn** – Provides neural network layers and modules.
* **torch.optim** – Provides optimization algorithms such as Adam.
* **torchvision** – Provides image datasets and image-processing utilities.
* **Matplotlib** – Used for displaying generated images.
* **NumPy** – Used for numerical and image-array operations.
* **CUDA** – Allows the model to use a compatible GPU when available.

The `device` variable automatically selects either GPU or CPU.

---

# STEP 2: DEFINING IMAGE TRANSFORMATIONS

The CIFAR-10 images are converted into tensors and normalized between **-1 and 1**.

### PROGRAM

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

## DESCRIPTION

### ToTensor()

Converts the input images into PyTorch tensors.

### Normalize()

The RGB channels are normalized using a mean of 0.5 and standard deviation of 0.5.

This changes the approximate pixel range from:

```text
0 – 1  →  -1 – 1
```

This normalization is suitable because the Generator uses the **Tanh** activation function at its output.

---

# PAGE 3 – STEP 3: LOADING THE CIFAR-10 DATASET

## STEP 3: LOADING THE CIFAR-10 DATASET

The CIFAR-10 training dataset is downloaded and loaded using torchvision.

### PROGRAM

```python
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

## DESCRIPTION

The `CIFAR10()` function downloads and loads the training dataset.

The `DataLoader` divides the dataset into mini-batches.

### PARAMETERS

| Parameter       |       Value |
| --------------- | ----------: |
| Dataset         |    CIFAR-10 |
| Training Images |      50,000 |
| Batch Size      |          32 |
| Shuffle         |        True |
| Image Size      | 32 × 32 × 3 |

Shuffling the dataset ensures that the model does not always see the training samples in the same order.

---

# STEP 4: DEFINING GAN HYPERPARAMETERS

Important training parameters are defined before constructing the GAN.

### PROGRAM

```python
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
```

## DESCRIPTION

### Latent Dimension

`latent_dim = 100` specifies the size of the random noise vector supplied to the Generator.

### Learning Rate

`lr = 0.0002` controls the step size used by the optimizers during training.

### Beta Parameters

`beta1` and `beta2` are parameters used by the Adam optimizer.

### Number of Epochs

`num_epochs = 10` specifies that the entire training dataset is processed ten times.

---

# PAGE 4 – STEP 5: BUILDING THE GENERATOR

## STEP 5: BUILDING THE GENERATOR

The Generator converts a random noise vector into a synthetic image.

### PROGRAM

```python
class Generator(nn.Module):

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(
                latent_dim,
                128 * 8 * 8
            ),

            nn.ReLU(),

            nn.Unflatten(
                1,
                (128, 8, 8)
            ),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128,
                momentum=0.78
            ),

            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64,
                momentum=0.78
            ),

            nn.ReLU(),

            nn.Conv2d(
                64, 3,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img
```

## GENERATOR WORKING

The Generator performs the following operations:

1. Accepts a 100-dimensional random noise vector.
2. Projects the noise into a larger feature representation.
3. Reshapes the representation into feature maps.
4. Upsamples the feature maps.
5. Applies convolutional layers to refine image features.
6. Uses Batch Normalization to improve training stability.
7. Uses ReLU activation for non-linear learning.
8. Produces a 3-channel RGB image.
9. Uses Tanh to produce output values in the range **-1 to 1**.

---

# PAGE 5 – STEP 6: BUILDING THE DISCRIMINATOR

## STEP 6: BUILDING THE DISCRIMINATOR

The Discriminator is a binary classifier that determines whether an image is real or generated.

### PROGRAM

```python
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.25),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ZeroPad2d(
                (0, 1, 0, 1)
            ),

            nn.BatchNorm2d(
                64,
                momentum=0.82
            ),

            nn.LeakyReLU(0.25),

            nn.Dropout(0.25),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                128,
                momentum=0.82
            ),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.25),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(
                256,
                momentum=0.8
            ),

            nn.LeakyReLU(0.25),

            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(
                256 * 5 * 5,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
```

## DISCRIMINATOR WORKING

* Convolutional layers extract features from images.
* Strided convolutions reduce spatial dimensions.
* LeakyReLU provides non-linear activation.
* Dropout helps reduce overfitting.
* Batch Normalization improves training stability.
* Flatten converts feature maps into a vector.
* Sigmoid produces a probability between 0 and 1.

The output represents the probability that the input image is real.

---

# PAGE 6 – STEP 7: INITIALIZING GAN COMPONENTS

## STEP 7: INITIALIZING GAN COMPONENTS

The Generator and Discriminator are initialized and moved to the selected device.

### PROGRAM

```python
generator = Generator(latent_dim).to(device)

discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=lr,
    betas=(beta1, beta2)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=lr,
    betas=(beta1, beta2)
)
```

## DESCRIPTION

### Generator

The Generator creates synthetic images from random noise.

### Discriminator

The Discriminator determines whether an image is real or fake.

### Binary Cross-Entropy Loss

`BCELoss()` is used because the Discriminator performs binary classification.

### Adam Optimizer

Separate Adam optimizers are used for:

* Generator
* Discriminator

This allows both networks to update their parameters independently.

---

# PAGE 7 – STEP 8: TRAINING THE GAN

## STEP 8: TRAINING THE GAN

During training, the Discriminator first learns to distinguish real and fake images. Then, the Generator is updated to produce images that can fool the Discriminator.

### PROGRAM

```python
for epoch in range(num_epochs):

    for i, batch in enumerate(dataloader):

        real_images = batch[0].to(device)

        valid = torch.ones(
            real_images.size(0),
            1,
            device=device
        )

        fake = torch.zeros(
            real_images.size(0),
            1,
            device=device
        )

        # ---------------------
        # Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        z = torch.randn(
            real_images.size(0),
            latent_dim,
            device=device
        )

        fake_images = generator(z)

        real_loss = adversarial_loss(
            discriminator(real_images),
            valid
        )

        fake_loss = adversarial_loss(
            discriminator(fake_images.detach()),
            fake
        )

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------

        optimizer_G.zero_grad()

        gen_images = generator(z)

        g_loss = adversarial_loss(
            discriminator(gen_images),
            valid
        )

        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
```

## TRAINING PROCESS

### Step 1

Real images are obtained from CIFAR-10.

### Step 2

Random noise vectors are generated.

### Step 3

The Generator creates fake images.

### Step 4

The Discriminator evaluates both real and fake images.

### Step 5

The Discriminator is updated using its loss.

### Step 6

The Generator is updated to make its generated images appear real.

This process is repeated throughout the training epochs.

---

# PAGE 8 – VISUALIZATION AND CONCLUSION

## VISUALIZING GENERATED IMAGES

Generated images can be displayed after training.

### PROGRAM

```python
if (epoch + 1) % 10 == 0:

    with torch.no_grad():

        z = torch.randn(
            16,
            latent_dim,
            device=device
        )

        generated = generator(z).detach().cpu()

        grid = torchvision.utils.make_grid(
            generated,
            nrow=4,
            normalize=True
        )

        plt.imshow(
            np.transpose(
                grid,
                (1, 2, 0)
            )
        )

        plt.axis("off")
        plt.show()
```

## OBSERVATION

The generated images are initially random and unclear. As GAN training progresses, the Generator learns features from the CIFAR-10 dataset and produces images that increasingly resemble the training data.

The quality of generated images depends on factors such as:

* Number of training epochs
* Generator architecture
* Discriminator architecture
* Learning rate
* Batch size
* Training stability

## RESULT

The GAN was successfully implemented using **PyTorch**. The Generator and Discriminator were trained adversarially using the CIFAR-10 dataset. The Generator learned to produce synthetic images from random noise, while the Discriminator learned to distinguish between real and generated images.

## CONCLUSION

Thus, a **Generative Adversarial Network (GAN)** was successfully implemented and trained using PyTorch. The experiment demonstrated the adversarial learning process between the Generator and Discriminator. The Generator learned to generate realistic-looking CIFAR-10 images, while the Discriminator evaluated whether the images were real or fake.

This experiment provides a basic understanding of GAN architecture, adversarial training, image generation, loss functions, and deep learning-based generative models.
