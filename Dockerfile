FROM python:3.9

RUN apt-get -y update && apt-get -y  upgrade && apt-get -y install supervisor
#RUN pip install supervisor-stdout

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app /app
COPY protos/* /app/protos/

COPY run_codegen.py /app/

WORKDIR /app
RUN python3 run_codegen.py

EXPOSE 8080 9090

ENTRYPOINT ["/usr/bin/supervisord"]
# ENTRYPOINT ["python3"]
# CMD ["gRPC_server.py", "http_server.py"]
# CMD ["http_server.py"]
