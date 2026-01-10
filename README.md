# Sketch-Search

Sketch-Search is a web application for:

- sketch-based image retrieval (sketch -> images),
- content-based image retrieval (image -> images).

The system computes vector embeddings for images and sketches and performs similarity search over an indexed gallery. It keeps the state of the database of indexed images up to date with filesystem.
It listens to changes made during server's runtime and does a reconciliation during startup.

## Installation

Clone the repository:

```bash
git clone https://github.com/mockoocy/sketch-search.git
cd sketch-search
```

### Running with Docker Compose

```bash
docker compose build
docker compose --profile app up
```

### Running locally

Frontend can be started with a Vite dev server using:

```bash
cd ui
pnpm install
pnpm run dev
```

Server-side app can be started using uv:

```bash
uv sync
uv run server
```

## Configuration

Server app can be configured using a YAML file.
It's contents are specified in server/src/server/config/models.py file.

Path to the config file can be passed as a command line argument when starting the server:

```bash
uv run server --config dev-config.yaml
```

### Adding own ML models for embedding generation.

For model to be registered, it has to be enclosed in class that implements method
with such signature.

```python
import numpy as np
import numpy.typing as npt
def embed(self, images: npt.NDArray[np.float32]) -> npt.NDArray[np.floating]:
```

The configuration is Deep Learning Framework-agnostic.

These can be registered using the YAML config, like so:

```yaml
embedder_registry:
  chosen_embedder: my_model
  embedders:
    my_model:
      target: server.embedders.default.DefaultEmbedder
      kwargs:
        weights_path: /weights/my_model.pth
    my_model2:
      file: /some/path/model.py
      class_name: MyModel
      args: ["arg1", 2]
      kwargs:
        weights_path: /weights/my_custom.pth
        some_flag: true
```

Example config for dev server with model passed using a dotted (python module) path:

```yaml
dev: true
embedder_registry:
  chosen_embedder: model_large
  embedders:
    model_large:
      target: server.embedders.default.DefaultEmbedder
      kwargs:
        weights_path: /weights/convnext_base_mixed.pth
        timm_backbone: convnext_base.fb_in22k_ft_in1k
    model_medium:
      target: server.embedders.default.DefaultEmbedder
      kwargs:
        weights_path: /weights/convnext_tiny_phase_1.pth
        timm_backbone: "convnext_tiny.fb_in22k"
    model_small:
      target: server.embedders.default.DefaultEmbedder
      kwargs:
        weights_path: /weights/efficientnet_phase_1.pth
        timm_backbone: efficientnet_b0.ra_in1k
```

### Using pre-trained ML models

Weights for the config above can be downloaded from [there](https://huggingface.co/datasets/Mockini/sbir/tree/main).
The config above works with the provided `docker-compose.yml` config if the downloaded weights are stored in `./weights` path (relative to the repository) root.

### Specifying filesystem directory to watch

Directory to watch can be specified using this config-file snippet.:

```yaml
watcher:
  watch_recursive: true
  watched_directory: /images
  files_batch_size: 64
```

### Specyifying config options using environment variables

Instead of using yaml config, config options can be specified using environment variables.
These can be used along YAML-based config (options coming from environment variables take precedence).

For example this snippet:

```yaml
database:
  database: db
  user: postgres
  password: password
```

could be expressed by setting values for `SERVER__DATABASE__DATABASE`, `SERVER__DATABASE__USER` and `SERVER__DATABASE__PASSWORD`, which may be more convenient in some deployment scenarios.

### Authorization

There are two authorization schemas available:

1. No auth
2. OTP auth

For the latter one has to specify credentials to some smtp server like so:

```yaml
auth:
  kind: otp
  smtp:
    host: "smtp.server"
    port: 576
    use_tls: true
    password: ...
    from_address: ...
  default_user_email: ...
```

The `auth.default_user_email` key specifies email of a user who's created during server startup
in case there are no users.

> [!NOTE]
> Providing config is optional. The app will work fine without any config (in no-auth mode).

## Additional remarks for using GPU

To make installation more flexible, no CUDA index is specified for the PyTorch dependencies.
So one has to install them additionally, e.g. with:

```bash
uv pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu128
```

Using Nvidia GPUs in containers can also be relatively trick.
For that purpose make sure that you have NVIDIA Container Toolkit installed.
