# Car Number Recognition

### How to install and test
1. Clone the repo. There are 5 test images in *_data_* directory, you can put there your own images.
```bash
git clone https://github.com/annabalabaeva/car-numbers-recognition
```
2. Create docker image and docker container. Choose nvidia or cpu version.
```bash
cd car-numbers-recognition/docker/cpu
```
or
```bash 
cd docker/nvidia
```
```bash
docker build -t balabaeva_car_number .
docker run -d -t -v $PWD/../../:/workdir --name car_number_recognition  balabaeva_car_number
docker exec -it car_number_recognition bash
```
3. Download the model
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PJSivxkmCr4GNAcvswq2xr7wnDhL342W' -O model/model.pth
```
4. Run test. Result will be printed in the console.
```bash
cd car_number_recognition
python -m ocr.predictor --data-dir=../data/ --model=../model/model.pth
```
