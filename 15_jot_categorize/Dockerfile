FROM ubuntu:18.04

RUN apt-get --yes update --fix-missing
RUN apt-get --yes install default-jre
RUN apt-get --yes install python3-pip --fix-missing 

# Application
RUN pip3 install --no-cache-dir fasttext
RUN pip3 install --no-cache-dir tqdm
RUN pip3 install --no-cache-dir -U scikit-learn
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir "dask[dataframe]" --upgrade

COPY ./py_app /code
COPY ./common_py /code
COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]

