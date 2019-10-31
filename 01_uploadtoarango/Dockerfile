FROM ubuntu:16.04

RUN apt-get update
RUN apt-get --yes --fix-missing install curl 

COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]
