from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import ast
import re

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

df = pd.read_csv("cleaned_recipes.csv")
df["ingredients"] = df["ingredients"].fillna("N/A")
df["instructions"] = df["instructions"].fillna("N/A")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

def retrieve_recommendation(query, top_k=10):
    recs = db.similarity_search(query, k=50)
    titles = [doc.metadata.get("title", "") for doc in recs]
    filtered_df = df[df["title"].isin(titles)].head(top_k)

    # Parse the ingredients and instructions into readable formats
    def parse_ingredients(ingredients):
        if isinstance(ingredients, str):
            try:
                ingredients = ast.literal_eval(ingredients)
            except:
                ingredients = [ingredients]
        return "\n".join([f"- {item}" for item in ingredients])

    def parse_instructions(instructions):
        if isinstance(instructions, str):
            try:
                instructions = ast.literal_eval(instructions)
            except:
                instructions = [instructions]
        
        def filter_and_format_instruction(text):
            nutrition_patterns = [
                r"Each \([^)]+\) serving has:.*",
                r"Nutrition Facts:.*",
                r"Per Serving:.*",
                r"Calories: \d+.*",
                r"Protein: \d+g.*",
                r"Carbohydrates: \d+g.*",
                r"Carbohydrate: \d+g.*",
                r"Total fat: \d+g.*",
                r"Total Fat: \d+g.*",
                r"Fat: \d+g.*",
                r"Saturated fat: \d+g.*",
                r"Cholesterol: \d+mg.*",
                r"Fiber: \d+g.*", 
                r"Sodium: \d+mg.*",
                r"\d+ calories.*",
                r"\d+g? (protein|carbs|fat).*",
                r"Nutritional Information.*",
                r"Nutrition Information.*"
            ]
            
            #filter nutritional info
            if isinstance(text, str):
                for pattern in nutrition_patterns:
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
                
                #remove numeric nutritional entries
                text = re.sub(r'\"\d+\":\s*\"(Calories|Protein|Carbohydrates|Fat|Saturated fat|Cholesterol|Fiber|Sodium):[^\"]*\"', "", text)
                
                # If the entire instruction is just a parenthetical note, we should exclude it
                if re.match(r'^\s*\([^)]+\)\s*$', text):
                    return ""
                
                text = re.sub(r'^\s*\(([^)]+)\)\s*(.+)$', r'\2 (\1)', text)
                
                return text.strip()
            return text
        
        if isinstance(instructions, dict) and all(isinstance(k, (str, int)) and str(k).isdigit() for k in instructions.keys()):
            # Create a new dictionary without nutritional information entries
            filtered_dict = {}
            for k, v in instructions.items():
                if isinstance(v, str):
                    #skip nutritional information
                    if not re.search(r'(Calories|Protein|Carbohydrates|Fat|Saturated fat|Cholesterol|Fiber|Sodium):', v, re.IGNORECASE) and not re.search(r'Each \([^)]+\) serving has:', v, re.IGNORECASE):
                        # Format parenthetical instructions and add if not empty
                        formatted_text = filter_and_format_instruction(v)
                        if formatted_text:  # Only add non-empty instructions
                            filtered_dict[k] = formatted_text
                else:
                    filtered_dict[k] = v
            
            instructions = filtered_dict
        
        # Apply filter based on data structure
        if isinstance(instructions, dict):
            filtered_instructions = {step: filter_and_format_instruction(text) for step, text in instructions.items()}
            # Filter out empty instructions
            filtered_instructions = {step: text for step, text in filtered_instructions.items() if text}
            return "\n".join([f"{step}. {text}" for step, text in filtered_instructions.items()])
        elif isinstance(instructions, list):
            filtered_instructions = [filter_and_format_instruction(text) for text in instructions]
            filtered_instructions = [text for text in filtered_instructions if text]
            return "\n".join([f"{i+1}. {text}" for i, text in enumerate(filtered_instructions)])
        else:
            return filter_and_format_instruction(str(instructions))
    
    recipes = []
    for _, row in filtered_df.iterrows():
        recipe = {
            "title": row["title"],
            "ingredients": parse_ingredients(row["ingredients"]),
            "instructions": parse_instructions(row["instructions"]),
            "image_filename": row.get("image_filename", "")
        }
        recipes.append(recipe)
    return recipes

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route("/", methods=["GET", "POST"])
def index():
    result = []
    caption = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        image_file = request.files.get("image")

        if image_file and image_file.filename != "":  # Image uploaded
            image = Image.open(image_file).convert('RGB')
            caption = generate_caption(image)
            result = retrieve_recommendation(caption)
        elif query:  # Text query entered
            result = retrieve_recommendation(query)

    return render_template("index.html", result=result, caption=caption)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
