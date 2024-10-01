import argparse

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from donor import predict_rotate


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-detect")
def process_request(file: UploadFile):
    image_bytes = file.file.read()
    result = predict_rotate(image_bytes)
    return StreamingResponse(result, media_type="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="127.0.0.1", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
