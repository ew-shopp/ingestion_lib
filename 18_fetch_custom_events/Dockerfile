ROM openjdk:8-jre-alpine

RUN apk add --update bash && rm -rf /var/cache/apk/*

RUN mkdir code
ADD *.jar /code/app.jar
COPY ./bash_scripts /code
COPY ./common_scripts /code

ENTRYPOINT ["/code/main.sh"]


