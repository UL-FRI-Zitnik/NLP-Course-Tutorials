# Tutorials for the Natural language processing course (UL FRI)
<sup>This repository is a part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) or [ales.zagar@fri.uni-lj.si](mailto:ales.zagar@fri.uni-lj.si)  for any comments.</sub>

# Project setup
** NOTE: Project setup is being updated weekly in the summer semester 2024/25. **

## Build your own image and run it

1. Build image

```
docker buildx build \
  --platform linux/amd64 \
  -t nlp-course-fri \
  ./docker_environment
```

2. Run container

```
docker run --platform linux/amd64 -it \
  --mount type=bind,source=$(pwd),target=/jupyter-data \
  -p 8888:8888 \
  nlp-course-fri
```

3. Navigate to http://localhost:8888 (password is 'Geslo.01') and enjoy!