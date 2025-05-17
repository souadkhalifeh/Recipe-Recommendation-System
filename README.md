# Recipe-Recommendation-System

## Project Overview
"When your fridge speaks in leftovers, let AI be your chef."

This intelligent system helps users answer the daily question: “What should I cook today?” Whether you're short on time, lacking cooking inspiration, or trying to reduce food waste, our app has you covered. Simply upload an image or type in your ingredients, and get delicious recipe suggestions tailored to your input.

## Problem Statement
Many people struggle with:
  - Deciding what to cook
  - Time constraints and busy schedules
  - Limited cooking skills
  - Wasting unused ingredients
  - Dietary restrictions

Our system bridges this gap using AI to offer instant, personalized cooking suggestions( in future).

## Methodology 
### 1. Input Handling
Accepts image uploads or text queries (e.g., “chicken, garlic, pasta”).

### 2. Image Understanding (BLIP Model)
Uses BLIP (Bootstrapped Language Image Pretraining) to generate captions from uploaded food images.

### 3. Recipe Retrieval
Text or image captions are transformed into vector embeddings using HuggingFace Transformers.
LangChain + ChromaDB power the vector search for relevant recipes in our dataset.

### 4. Recipe Matching
Top 10 matched recipes are selected from a cleaned Kaggle dataset of food and ingredients.

### 5. Output Formatting
Ingredients and instructions are parsed, cleaned, and structured for easy reading.

### 6. Display
Recipes are shown in card format with:
  - Title
  - Ingredients
  - Instructions

## Features
- Multi-modal interface: text or image-based queries
- Uses BLIP, HuggingFace, and LangChain
- Powered by Flask, HTML/CSS frontend
- Returns 10 relevant, easy-to-follow recipes

## Tech Stack 
  - Frontend : HTML, CSS
  - Backend : Flask (Python)
  - AI Models : BLIP, HuggingFace Embeddings
  - Retrieval Engine : LangChain + ChromaDB
  - Dataset : Kaggle Food Ingredients Dataset (https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
    
## How to run this model
  -docker build -t recipe_app
  -docker run -p 5000:5000
  
## Future Enhancements
  - Personalized user profiles (diet, allergies, preferences)
  - Mobile app version
  - Multi-language support (Arabic, French, etc.)
