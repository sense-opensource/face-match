
## Introduction
Sense Face Match leverages state-of-the-art deep learning models to deliver highly accurate face detection, embedding extraction, and face comparison. Designed for both developers and enterprises, it supports secure user authentication, and face match checks —all while keeping biometric data private and local.

Whether you are building secure onboarding for fintech or adding face login to your app, Sense Face Match provides the tools and flexibility you need.

### Install Python Dependencies
pip install -r requirements.txt

### Start the FastAPI Server
uvicorn app:app --reload

This will start the API server on:
http://localhost:3015


### Running with Docker
### Build Docker Image
docker build -t sense_face_verification_image .

### Run Docker Container
docker run -d --name sense_face_verification_container -p 3015:3015 sense_face_verification_image

This will start the API server on:
http://localhost:3015


### 4. Run the Frontend

cd front-end
npm install
npm run dev

By default, the frontend runs on:
http://localhost:3010


### Useful Docker Commands

# Stop container
docker stop sense_face_verification_container

# Remove container
docker rm -f sense_face_verification_container

# Remove image
docker rmi -f sense_face_verification_image

# View logs
docker logs sense_face_verification_container


### License
MIT License — free to use, share, and modify.

### Demo url
https://getsense.co/face-match/
