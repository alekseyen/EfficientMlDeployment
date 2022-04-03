FROM python:3.9

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ADD app /app

WORKDIR /app

ENTRYPOINT ["python3", "http_server.py"]
