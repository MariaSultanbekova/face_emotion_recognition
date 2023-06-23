from fastapi import FastAPI, UploadFile, File
import torch
from torchvision.transforms import transforms
import torchvision.models as models
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO


app = FastAPI(title='emotion_classifier')


origins = [
    "http://localhost:3000",  # для тестирования
    "http://ai-project-21.ru"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", 'DELETE', 'PUT', 'PATCH'],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin",
                   "Authorization"],
)



model = models.resnet34()
num_classes = 7
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/emotional_classificator.pt', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post('/predict_emotion')
async def predict_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    image = transform(image)

    with torch.no_grad():
        image = image.unsqueeze(0)
        prediction = model(image)
        _, predicted_idx = torch.max(prediction, 1)
        emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][predicted_idx.item()]

    return {"emotion": emotion}


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
