FROM ubuntu:18.04

RUN apt-get --yes update --fix-missing
RUN apt-get --yes install python3-pip --fix-missing 

COPY ./py_app /code
COPY ./common_py /code
COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]

