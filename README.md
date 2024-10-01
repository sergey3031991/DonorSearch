# DonorSearch
DonorS


Model for cetrificate №45  classification and and flip the uploaded images

Research Purposes

It is necessary to develop a model that determines degrees images and flip the uploaded images.Source Dataset
Classification

There were several models, such as ResNet50 with similar results (accuracy about 0.95, ROC-AUC about 1.0).



Run

clone the repo
download the models' weights to Desktop/dc:
ResNet
SSD
build the docker image
  docker build --tag swag .
run the container
  docker run --rm -it -p 8000:8000 --name app swag
the app is available at 8000 port

