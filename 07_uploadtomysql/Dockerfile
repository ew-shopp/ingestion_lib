FROM ubuntu:18.04

RUN apt-get --yes update
RUN apt-get --yes install mysql-client

COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]


