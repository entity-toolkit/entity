## Self-hosted runners 

GitHub allows to listen to repository changes and run the so-called "actions" (e.g., tests) if a particular event has been triggered (e.g., a pull request has been created). To test Entity on GPUs, we provide Docker runner images, with all the proper compilers already preinstalled, which can fetch the actions directly from GitHub and run them within the container environment.

To do that, one needs to create an image with the corresponding `Dockerfile`, and then launch a Docker container which will run in the background, listening to commands and running any actions forwarded from the GitHub.

### NVIDIA GPUs

```sh
# 1. Create the image
docker build -t ghrunner:nvidia -f Dockerfile.runner.nvidia .
# 2. Run a container from the image with GPU support 
# ... (see wiki for instructions on NVIDIA runtime)
docker run --runtime=nvidia --gpus=all -dt ghrunner:nvidia
```
