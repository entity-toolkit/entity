## Self-hosted runners 

GitHub allows to listen to repository changes and run the so-called "actions" (e.g., tests) if a particular event has been triggered (e.g., a pull request has been created). To test Entity on GPUs, we provide Docker runner images, with all the proper compilers already preinstalled, which can fetch the actions directly from GitHub and run them within the container environment.

To do that, one needs to create an image with the corresponding `Dockerfile`, and then launch a Docker container which will run in the background, listening to commands and running any actions forwarded from the GitHub.

First, you will need to obtain a runner token from the Entity GitHub repo, by going to Settings -> Actions -> Runners -> New self-hosted runner. Copy the token, and save it to a file:

```sh
echo ******TOKEN****** > action-token
```

Images differ slightly from the type of runner. 

### NVIDIA GPUs

```sh
docker build -t ghrunner:nvidia -f Dockerfile.runner.nvidia --secret id=ghtoken,src=./action-token .
docker run --runtime=nvidia --gpus=all -dt ghrunner:nvidia
```

### AMD GPUs

```sh
docker build -t ghrunner:amd -f Dockerfile.runner.rocm --secret id=ghtoken,src=./action-token .
docker run --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video -dt ghrunner:amd
```
