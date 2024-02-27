# Docker environment for running notebooks

Run all commands from the root of the repository.

## Run prebuilt image from Dockerhub

1. Run container

```
docker run --platform linux/amd64 -it \
  --env JUPYTER_DIR='/jupyter-data' \
  --env JUPYTER_PASSWORD='Geslo.01' \
  --mount type=bind,source=$(pwd),target=/jupyter-data \
  -p 8888:8888 \
  -d azagsam2468/nlp-course-fri-updated
```

2. Navigate to http://localhost:8888 and enjoy!

## Build your own image and run it (+ optionally publish to Dockerhub)

1. Build image

```
docker buildx build \
  --platform linux/amd64 \
  -t nlp-course-fri \
  -f ./docker_environment/Dockerfile \
  ./docker_environment
```

2. Run container

```
docker run --platform linux/amd64 -it \
  --env JUPYTER_DIR='/jupyter-data' \
  --env JUPYTER_PASSWORD='Geslo.01' \
  --mount type=bind,source=$(pwd),target=/jupyter-data \
  -p 8888:8888 \
  -d nlp-course-fri
```

3. Navigate to http://localhost:8888 and enjoy!

### Publishing image to Dockerhub

```
docker login
docker tag nlp-course-fri szitnik/nlp-course-fri:01
docker push szitnik/nlp-course-fri:01
```