FROM ubuntu:16.04

RUN apt-get --yes update
RUN apt-get --yes install python; python --version
RUN apt-get --yes install python-pip; pip --version
RUN pip install --upgrade pip

RUN apt-get --yes install build-essential python-dev
RUN apt-get --yes install wget
RUN pip install  pyopenssl ndg-httpsclient pyasn1


# Application
RUN pip install --no-cache-dir pandas

COPY ./py_app /code
COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]


