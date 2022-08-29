# base image
FROM continuumio/miniconda3:4.10.3p0-alpine

# container working directory
WORKDIR /code

# dependencies
COPY requirements.txt .

# install dependencies
RUN apk add --update-cache git \
    && apk add busybox=1.35.0-r22 --repository=http://dl-cdn.alpinelinux.org/alpine/edge/main \
    && pip install -r requirements.txt

# copy the src directory
COPY src/ .

RUN ./run_test.sh

# container start command
CMD [ "python", "./main.py" ]
