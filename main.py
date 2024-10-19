from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, Field
from transformers import pipeline
from fastapi.responses import JSONResponse
import os
import logging

# Set the cache directory for Hugging Face Transformers
os.environ['HF_HOME'] = '/tmp'  # Set the Hugging Face cache directory to /tmp

# Define the input data model with constraints
class LyricsInput(BaseModel):
    language: constr(min_length=1, max_length=20) = Field(..., description="The language of the song")
    genre: constr(min_length=1, max_length=20) = Field(..., description="The genre of the song")
    description: constr(min_length=1, max_length=100) = Field(..., description="A description of the song")  # Reduced to 100 chars

# Initialize FastAPI app
app = FastAPI()

# Load the model once at startup
lyric_generator = pipeline('text-generation', model='gpt2', pad_token_id=50256)

logging.basicConfig(level=logging.INFO)

# Create the lyrics generation endpoint
@app.post("/generate_lyrics")
async def generate_lyrics(input: LyricsInput):
    prompt = f"Write a {input.genre} song in {input.language} about {input.description}"

    # Ensure the total length of the prompt is reasonable
    if len(prompt) > 150:
        return JSONResponse(content={"error": "Prompt too long. Please shorten your input."}, status_code=400)

    # Generate lyrics
    try:
        generated_lyrics = lyric_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    except (ValueError, TypeError) as e:
        logging.error(f"Error generating lyrics: {str(e)}")
        return JSONResponse(content={"error": "Invalid input parameters."}, status_code=400)
    except Exception as e:
        logging.error(f"Unexpected error generating lyrics: {str(e)}")
        return JSONResponse(content={"error": "Failed to generate lyrics."}, status_code=500)

    # Truncate the output if it exceeds 300 characters
    if len(generated_lyrics) > 300:
        generated_lyrics = generated_lyrics[:300] + "... [truncated]"

    return {"lyrics": generated_lyrics}

# Define a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Music Production API"}

# Add a dummy favicon route
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={"message": "Favicon not found"})
