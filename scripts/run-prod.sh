#!/usr/bin/env bash

NAME=u2net

set -x

docker rm -f $NAME

docker run -d --restart always \
  --name $NAME \
  -v $HOME/.paddlehub:/root/.paddlehub \
  -v $PWD/saved_models/:/src/saved_models/ \
  -p 5007:5000 \
  --gpus all \
  r8.im/devxpy/$NAME

docker logs -f $NAME
