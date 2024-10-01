FROM python:3.12-bookworm
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY templates /app/templates
COPY src /app/src
COPY resnet_trained.pth /app/
CMD python src/app.py
