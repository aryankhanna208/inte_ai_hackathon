FROM docker.io/python:latest

WORKDIR /usr/src/app

COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8000

CMD ["python3", "api_testing.py"]
