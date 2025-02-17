from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__, template_folder="templates")

# Load trained model
model = tf.keras.models.load_model("skin_type_model.h5")

# Class labels
class_labels = ["Dry", "Normal", "Oily"]

# Skincare product recommendations
skincare_products = {
    "Oily": {
        "Face Wash": [
            {
                "name": "La Roche-Posay Effaclar Purifying Foaming Gel",
                "link": "https://www.laroche-posay.us/our-products/effaclar/effaclar-purifying-foaming-gel.html",
                "image": "https://m.media-amazon.com/images/I/61fD94+o7nL._SL1200_.jpg"
            },
            {
                "name": "Paula's Choice CLEAR Pore Normalizing Cleanser",
                "link": "https://www.paulaschoice.com/clear-pore-normalizing-cleanser/305.html",
                "image": "https://www.paulaschoice.com/dw/image/v2/BBNX_PRD/on/demandware.static/-/Sites-pc-catalog/default/dw1cc2db81/images/products/clear-pore-normalizing-cleanser-6001-portrait.png?sw=2000&sfrm=png"
            }
        ],
        "Moisturizer": [
            {
                "name": "Cetaphil Oil-Free Hydrating Lotion",
                "link": "https://www.cetaphil.com/us/product/oil-free-hydrating-lotion",
                "image": "https://i5.walmartimages.com/seo/Cetaphil-Daily-Oil-Free-Hydrating-Lotion-for-Face-with-Hyaluronic-Acid-3-oz_d6772746-cbc0-4e8f-b055-8972b26b9700.3376acaa52d5f89118e4a8a15ae6d1e1.jpeg"
            },
            {
                "name": "Neutrogena Oil-Free Moisturizer",
                "link": "https://www.neutrogena.com/products/skincare/oil-free-moisture-sensitive-skin.html",
                "image": "https://www.epharmacy.com.np/content/images/thumbs/61b5d45fc70315faaaef38d2_neutrogena-oil-free-moisture-sensitive-skin-118ml.jpeg"
            }
        ],
        "Toner": [
            {
                "name": "Paula’s Choice Skin Perfecting 2% BHA",
                "link": "https://www.paulaschoice.com/skin-perfecting-2-percent-bha-liquid-exfoliant/201.html",
                "image": "https://www.paulaschoice.com/dw/image/v2/BBNX_PRD/on/demandware.static/-/Sites-pc-catalog/default/dwb8a94748/images/products/2-percent-bha-liquid-exfoliant-2010-L.png?sw=2000&sfrm=png"
            },
            {
                "name": "The Ordinary Glycolic Acid 7% Toning Solution",
                "link": "https://theordinary.deciem.com/product/rdn-glycolic-acid-7pct-toning-solution-240ml",
                "image": "https://static-01.daraz.com.np/p/f112acc1b4a8ac4f1c30a402534d2127.jpg"
            }
        ]
    },
    "Dry": {
        "Face Wash": [
            {
                "name": "CeraVe Hydrating Facial Cleanser",
                "link": "https://www.cerave.com/skincare/cleansers/hydrating-cleanser",
                "image": "https://i.imgur.com/CeraVeHydratingCleanser.jpg"
            },
            {
                "name": "Neutrogena Hydro Boost Hydrating Cleansing Gel",
                "link": "https://www.neutrogena.com/products/skincare/hydro-boost-hydrating-cleansing-gel.html",
                "image": "https://i.imgur.com/NeutrogenaHydroBoost.jpg"
            }
        ],
        "Moisturizer": [
            {
                "name": "CeraVe Moisturizing Cream",
                "link": "https://www.cerave.com/skincare/moisturizers/moisturizing-cream",
                "image": "https://www.epharmacy.com.np/content/images/thumbs/668bb2724b7053e4c9017317_cerave-hydrating-facial-cleanser-473ml.jpeg"
            },
            {
                "name": "Eucerin Advanced Repair Cream",
                "link": "https://www.eucerinus.com/products/advanced-repair/cream",
                "image": "https://images-1.eucerin.com/~/media/eucerin%20relaunch%20media/media-center-items/d/6/f/f484206c652c49eab0ba3e713b85f0b9-screen.jpg"
            }
        ],
        "Toner": [
            {
                "name": "Klairs Supple Preparation Unscented Toner",
                "link": "https://www.klairscosmetics.com/product/unscented-toner/",
                "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQy6O2yliKtt7DHOzYLR8t1tqzIGYXIFqxFeg&s"
            },
            {
                "name": "Hada Labo Gokujyun Hydrating Toner",
                "link": "https://www.hadalabo.com.my/products/gokujyun-lotion",
                "image": "https://sukoshimart.com/cdn/shop/products/hada-labo-gokujyun-toner-main_1024x1024.png?v=1647936894"
            }
        ]
    }
}

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded.", recommendation={})

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", result="No selected file.", recommendation={})

        # Save uploaded file
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        # Process image and predict
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]

        # Get skincare recommendation
        recommendation = skincare_products.get(predicted_class, {})

        return render_template("index.html", result=predicted_class, recommendation=recommendation)

    return render_template("index.html", result=None, recommendation={})

if __name__ == "__main__":
    app.run(debug=True)
