from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import json
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

# Recipe retrieval function
def retrieve_recommendation(query, top_k=10):
    recs = db.similarity_search(query, k=50)
    titles = [doc.metadata.get("title", "") for doc in recs]
    filtered_df = df[df["title"].isin(titles)].head(top_k)
    return filtered_df[["title", "ingredients", "instructions"]].to_dict(orient="records")

# Caption generation using BLIP
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

        if image_file and image_file.filename != "":
            image = Image.open(image_file).convert('RGB')
            caption = generate_caption(image)
            result = retrieve_recommendation(caption)

        elif query:
            result = retrieve_recommendation(query)

    return render_template("index.html", result=result, caption=caption)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
