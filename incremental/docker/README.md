# Run In Docker
## Steps
### 1. Change mode
```bash
chmod -R 777 tensorflow-get-started/
```
### 2. Change directory to `tensorflow-get-started/incremental/docker`
```bash
sh run_docker.sh $CONTAINNER_NAME $GPU_ID
```
- For example
```bash
sh run_docker.sh r10922176 0
```
### 3. Attach to container
```bash
docker attach $CONTAINNER_NAME
```
### 4. CD to `/home/docker/project` in container and run.
```bash
cd /home/docker/project
sh run.sh
```