# Create a virtual environment

```sh
virtualenv --python=/home/torak28/.pyenv/versions/3.7.0/bin/python3.7 .venv
```

# Get inside the environment

```sh
source .venv/bin/activate
```

# Install ipykernel

```sh
python -m pip install ipykernel
```

# Install a Jupyter kernel

```sh
ipython kernel install --user --name=.venv
```

# Check ```kernel.json```

```sh
code /home/torak28/.local/share/jupyter/kernels/.venv/kernel.json
```

if need change *argv* to ```/home/torak28/Desktop/Studia/Magisterka/Image-manipulation-detection/.venv/bin/python```

# Launch

```sh
jupyter-notebook
```