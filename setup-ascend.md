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

## Start tmux

```bash
tmux
```

## Start Jupyter Notebook Server

To start Jupyter Notebook Server, run the following command:

```bash
jupyter notebook --no-browser
```

Please note the port number of the Jupyter Server running on.

## Detach from tmux session

Press `Ctrl+B` then `d` to detach from `tmux`. After that, we will need to disconnect from server by `exit` command.

```bash
exit
```

## Reconnect to server with port forwarding

```bash
ssh -L 8888:localhost:8888 your_department_login@172.27.244.41
```

Please replace `8888` with the port number that your Jupyter Server running on.

## Reatach to previous terminal session

```bash
tmux attach
```

# Troubleshooting

## DataSpell `updating python interpreter...`

In Menu `Help` -> `Diagnostics Tool` -> ``, add `#com.jetbrains.python.sdk.PythonSdkUpdater$Trigger` as restart DataSpell
