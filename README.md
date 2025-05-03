
### Install Python Dependencies
pip install -r requirements.txt

### Start the FastAPI Server
uvicorn app:app --reload

This will start the API server on:
http://localhost:8000


### Running with Docker
### Build Docker Image
docker build -t sense_face_verification_image .

### Run Docker Container
docker run -d --name sense_face_verification_container -p 8000:8000 sense_face_verification_image

This will start the API server on:
http://localhost:8000


### 4. Run the Frontend

cd front-end
npm install
npm start

By default, the frontend runs on:
http://localhost:3000


### Project Structure
.
├── Dockerfile
├── docker-compose.yml
├── app.py/              # FastAPI app entrypoint
├── src/                 # Anti-spoofing model logic
├── resources/           # Pretrained model files
└── front-end/           # Frontend application (optional)


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


