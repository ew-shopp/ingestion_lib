FROM python:3.7
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r /code/requirements.txt
COPY ./py_app /code
COPY ./common_py /code
COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]

