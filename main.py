from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from TTS.api import TTS
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Configure Tesseract OCR for multiple languages
ocr_languages = "eng+hin+mar"

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else "Error: No text extracted. Try an image-based PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=ocr_languages)
        return text.strip() if text.strip() else "Error: No text found in image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def generate_cloned_speech(speaker_audio_path, input_text, output_path):
    try:
        tts.tts_to_file(
            text=input_text,
            file_path=output_path,
            speaker_wav=speaker_audio_path,
            language="en",
            split_sentences=True
        )
        return True
    except Exception as e:
        return str(e)

@app.post("/process/")
async def process_files(speaker_audio: UploadFile = File(...), input_file: UploadFile = File(...)):
    try:
        # Save uploaded files temporarily
        speaker_audio_path = f"/tmp/{speaker_audio.filename}"
        input_file_path = f"/tmp/{input_file.filename}"
        output_audio_path = "/tmp/output.wav"

        with open(speaker_audio_path, "wb") as buffer:
            shutil.copyfileobj(speaker_audio.file, buffer)
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(input_file.file, buffer)

        # Determine input file type and extract text
        file_extension = os.path.splitext(input_file.filename)[1].lower()
        if file_extension == ".pdf":
            input_text = extract_text_from_pdf(input_file_path)
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            input_text = extract_text_from_image(input_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF, JPG, JPEG, or PNG file.")

        if not input_text or "Error" in input_text:
            raise HTTPException(status_code=400, detail=f"Error: {input_text}")

        # Generate speech
        speech_result = generate_cloned_speech(speaker_audio_path, input_text, output_audio_path)
        if speech_result is not True:
            raise HTTPException(status_code=500, detail=f"Error during speech generation: {speech_result}")

        return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7600))
    uvicorn.run(app, host="0.0.0.0", port=port)
