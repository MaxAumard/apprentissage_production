FROM python:3.11-slim-buster
LABEL authors="Max"
WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
COPY . .
CMD ["/start.sh"]