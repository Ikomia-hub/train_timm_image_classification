<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_timm_image_classification/main/icons/timm.png" alt="Algorithm icon">
  <h1 align="center">train_timm_image_classification</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_timm_image_classification">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_timm_image_classification">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_timm_image_classification/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_timm_image_classification.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train timm image classification models.

![Rock paper scissors](https://uploads-ssl.webflow.com/645cec60ffb18d5ebb37da4b/64e480470f4a9d7b0a3198fb_Picture23-p-800.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
data_loader = wf.add_task(name="dataset_classification")

data_loader.set_parameters({
    "dataset_folder": "path/to/dataset/folder",
}) 

train = wf.add_task(name="train_timm_image_classification", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'resnet18': Name of the pre-trained model. 
    - There are over 700 timm models. You can list them using: timm.list_models()
- **input_size** (list) - default '[224, 224]': Size of the input image.
- **epochs** (int) - default '100': Number of complete passes through the training dataset.
- **batch_size** (int) - default '16': Number of samples processed before the model is updated.
- **learning_rate** (float) - default '0.0050': Step size at which the model's parameters are updated during training.
- **output_folder** (str, *optional*): path to where the model will be saved. 


**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
data_loader = wf.add_task(name="dataset_classification")

data_loader.set_parameters({
    "dataset_folder": "C:/Users/allan/OneDrive/Desktop/ik-desktop/Images/datasets/Fruit",
}) 

# Add train algorithm 
train = wf.add_task(name="train_timm_image_classification", auto_connect=True)
train.set_parameters({
    "model_name": "resnet34",
    "batch_size": "8",
    "epochs": "5",
    "learning_rate": "0.0050",
}) 
# Launch your training on your data
wf.run()
```

