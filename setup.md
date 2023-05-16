# Pre-Lab1 Python Virtual Environment and Deep Learning Frameworks

---

In this lab, you will learn how to create a virtual environment and install deep learning frameworks, such as [**TensorFlow**](https://www.tensorflow.org/), [**PyTorch**](https://pytorch.org/), and [**Mindspore**](https://www.mindspore.cn/en). You will also learn how to install [**Jupyter Notebook**](https://jupyter.org/) and [**JupyterLab**](https://jupyterlab.readthedocs.io/en/stable/). You will gain access to Github Student Developer Pack, which provides a lot of free tools for students, such as [**DataSpell**](https://www.jetbrains.com/dataspell/) and [**PyCharm**](https://www.jetbrains.com/pycharm/). Finally, you will also gain access to our department's Huawei Ascend 910 AI Server which supported by **Mindspore** natively.


# Overview for Tools, Modules and Deep Learning Frameworks

## Virtual Environment

Python Virtual Environment is a tool to keep the dependencies required by different projects in separate places, by creating virtual Python environments for them. It solves the “Project X depends on version 1.x but, Project Y needs 4.x” dilemma, and keeps your global site-packages directory clean and manageable. There are several tools to create virtual environments in Python. In this lab, we will use [**venv**](https://docs.python.org/3/library/venv.html) and [**conda**](https://conda.io/).

![](https://python-for-scientists.readthedocs.io/en/latest/_images/environments_folders-2.png)

### Conda
[**conda**](https://conda.io/) is a package manager, environment manager, and Python distribution that contains a lot of scientific packages. It is used to create virtual environments for Python containing both Python and non-Python packages. A **conda** environment is a directory that contains a specific collection of conda packages that you have installed even if you have multiple versions of Python on your computer, you can specify which Python installation each environment uses. For example, you may have one environment with NumPy 1.7 and its dependencies, and another environment with NumPy 1.6 for legacy testing. If you change one environment, your other environments are not affected. You can easily activate or deactivate environment, which is how you switch between them. You can also share your environment with someone by giving them a copy of your environment.yaml file.

### venv

**venv** is a built-in module in Python 3 for creating virtual environments. It was added to Python in version 3.3. It’s intended to be the successor of the older [**virtualenv**](https://virtualenv.pypa.io/) tool. **virtualenv** is a third party alternative (and predecessor) to **venv**. It allows you to create and manage separate environments with different versions of Python and/or packages installed in them.

## Deep Learning Frameworks

Deep Learning Frameworks are software libraries that are designed to make building deep learning models fast, easy, and efficient. They provide a clear and concise way for defining models using a collection of pre-built and optimized components. Deep learning frameworks are often optimized for distributed training across multiple GPUs and machines. They also provide a suite of visualization tools to make it easier to understand, debug, and optimize your models.

### TensorFlow

[**TensorFlow**](https://www.tensorflow.org/) is an open-source software library for machine learning across a range of tasks, and developed by Google to meet their needs for systems capable of building and training neural networks to detect and decipher patterns and correlations, analogous to the learning and reasoning which humans use. It is currently one of the most popular deep learning frameworks.

### PyTorch

[**PyTorch**](https://pytorch.org/) is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license. Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface.

### Mindspore

[**Mindspore**](https://www.mindspore.cn/en) is a deep learning framework developed by Huawei. It is designed to provide development experience with friendly design and efficient execution for deep learning algorithms, covering model architecture search, training and inference. Mindspore is adaptable to all scenarios that require AI technologies.


## Jupyter Notebook, JupyterLab and DataSpell

[**Jupyter Notebook**](https://jupyter.org/) is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.

[**JupyterLab**](https://jupyterlab.readthedocs.io/en/stable/) is the next-generation web-based user interface for Project Jupyter. JupyterLab enables you to work with documents and activities such as Jupyter notebooks, text editors, terminals, and custom components in a flexible, integrated, and extensible manner. You can arrange multiple documents and activities side by side in the work area using tabs and splitters. Documents and activities integrate with each other, enabling new workflows for interactive computing.

# Create a virtual environment using conda for Pytorch and Tensorflow

`conda` command is used to create, manage, and activate virtual environments. To create a virtual environment, run the following command:

```bash
conda create -n py39 python=3.9.16
```

The above command creates a virtual environment named `py39` with Python version 3.9.16. To activate the virtual environment, run the following command:

```bash
conda activate py39
```

## Install PyTorch

After that, you can install PyTorch using the following command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Install Tensorflow

```bash
python -m pip install -U tensorflow
```

## Install Jupyter and other packages

You can install Jupyter and other packages using the following command:

```bash
conda install jupyter matplotlib numpy pandas
```

## Install diffusion and transformers

```bash
python -m pip install -U diffusion transformers
```


# Setting up DataSpell

[**DataSpell**](https://www.jetbrains.com/dataspell/) is a powerful IDE for data science and machine learning. It is based on [**PyCharm**](https://www.jetbrains.com/pycharm/), which is one of the most popular Python IDEs. DataSpell provides a more powerful alternative to JupyterLab. It supports Jupyter Notebook, JupyterLab, and other popular data science tools. It also provides a lot of powerful features, such as code completion, code refactoring, version control, and so on.

## Activate license for DataSpell with Github Student Developer Pack

Let's apply for Github Student Developer Pack first. You can apply for it [here](https://education.github.com/pack). After you get the pack, you can activate DataSpell with the license provided by Github Student Developer Pack.

After that you could request a free subscription for DataSpell [here](https://www.jetbrains.com/shop/eform/students). You will receive an email with a link to activate your subscription. After you activate your subscription, you will receive an email with a license key. You can activate DataSpell with the license key.

## Configure DataSpell with existing conda environment

In DataSpell, you can configure the Python interpreter with existing conda environment. You can do this by clicking the Python interpreter in the status bar.

![](https://resources.jetbrains.com/help/img/idea/2023.1/py_ds_new_conda_env.png)

Please select the existing conda environment you created before.

## Attach directory to DataSpell

In DataSpell, you can attach a directory to the project. You can do this by clicking the `Attach Directory` button in the status bar.

![](https://resources.jetbrains.com/help/img/idea/2023.1/py_ds_add_existing_directory.png)

Please select the directory you want to attach.

## Create a new Jupyter Notebook

In DataSpell, you can create a new Jupyter Notebook by clicking the `+` button in the status bar.

![](https://resources.jetbrains.com/help/img/idea/2023.1/py_ds_new_notebook.png)

## Run a Jupyter Notebook

In DataSpell, you can run a Jupyter Notebook by clicking the `Run` button in the status bar.

![](https://resources.jetbrains.com/help/img/idea/2023.1/py_ds_run_actions.png)

## Verify PyTorch installation by running Stable Diffusion Model

Previously, we have installed PyTorch. Now we can verify the installation by running Stable Diffusion Model.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=20).images[0]

image.save("astronaut_rides_horse.png")
```

# Setting up venv for Tensorflow on Huawei Ascend Server

Our department has a Huawei Ascend Server with `mindspore` support. However, it does not support Tensorflow. In this section, we will show you how to set up venv for Tensorflow on Huawei Ascend Server. This server is only available for students in our department and will be used to run all lab experiments in this course.

## Connect to Huawei Ascend Server with SSH

You can connect to Huawei Ascend Server with SSH. You can do this by running the following command in your terminal:

```bash
ssh your_department_login@172.27.244.41
```

## Create a virtual environment using venv

`venv` module provides support for creating lightweight “virtual environments” with their own site directories, optionally isolated from system site directories. Each virtual environment has its own Python binary (allowing creation of environments with various Python versions) and can have its own independent set of installed Python packages in its site directories.

To create a virtual environment, run the following command:

```bash
/usr/local/python3.7.5/bin/python3 -m venv ~/py37
```

It will create a virtual environment named `py37` with Python version 3.7.5 in home directory. To activate the virtual environment, run the following command:

```bash
. ~/py37/bin/activate
```

## Upgrade pip

It's recommended to upgrade pip before installing any packages in the virtual environment. To do this, you can upgrade pip using the following command:

```bash
pip install -U pip
```

## Install Tensorflow

After that, you can install Tensorflow using the following command:

```bash
pip install -U tensorflow
```

## Install Jupyter Notebook, JupyterLab and other packages

After that, you can install Jupyter Notebook using the following command:

```bash
pip install -U jupyter matplotlib numpy pandas
```

## Start Jupyter Notebook Server

To start Jupyter Notebook Server, run the following command:

```bash
jupyter notebook --no-browser
```

# Troubleshooting

## DataSpell `updating python interpreter...`

In Menu `Help` -> `Diagnostics Tool` -> ``, add `#com.jetbrains.python.sdk.PythonSdkUpdater$Trigger` as restart DataSpell
