FROM ubuntu:16.04
#FROM python:2

RUN apt-get --yes update
RUN apt-get --yes install python; python --version
RUN apt-get --yes install python-pip; pip --version
RUN pip install --upgrade pip
#RUN easy_install -U pip; python --version; pip --version
#RUN sudo pip install virtualenv; virtualenv --version
RUN apt-get --yes install build-essential python-dev
RUN apt-get --yes install wget
RUN apt-get --yes install unzip
RUN pip install  pyopenssl ndg-httpsclient pyasn1

WORKDIR /weather
#RUN virtualenv weather-env

RUN pip install --no-cache-dir numpy 
RUN pip install --no-cache-dir https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz

# Install OpenWeatherMap request lib
RUN pip install --no-cache-dir requests

# Weather lib installation

COPY weather-data-master.zip /weather
WORKDIR /weather
RUN unzip weather-data-master.zip
WORKDIR /weather/weather-data-master
RUN python setup.py install


# Application
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir geopy 

COPY ./py_app /weather/weather-data-master/weather_import
COPY ./common_py /weather/weather-data-master/weather_import

COPY ./bash_scripts /code
COPY ./common_scripts /code

# Make rw access for all users
RUN chmod 777 /weather; \
    chmod 777 /weather/weather-data-master; \
    chmod 777 /weather/weather-data-master/weather_import

ENTRYPOINT ["/code/main.sh"]


