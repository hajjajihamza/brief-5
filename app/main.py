
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import traceback
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from pathlib import Path
from PIL import Image

from src.model import predict_image


# ==================================================
# 1. Chemins des fichiers
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "output"
    / "models"
    / "mobilenetv2_1_00_224"
    / "best_model_ft.keras"
)

ENCODER_PATH = BASE_DIR.parent / "label_encoder.pkl"


# ==================================================
# 2. Configuration Streamlit
# ==================================================

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroVision AI")
st.subheader("Classification des images IRM cérébrales")

st.write(
    "Téléchargez une image IRM pour tester "
    "la prédiction du modèle MobileNetV2."
)


# ==================================================
# 3. Chargement du modèle
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_label_encoder():
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError(
            f"LabelEncoder introuvable : {ENCODER_PATH}"
        )

    return joblib.load(ENCODER_PATH)


try:
    model = load_model()
    label_encoder = load_label_encoder()

    st.success("Modèle et LabelEncoder chargés avec succès.")

except Exception as error:
    st.error("Erreur lors du chargement des fichiers.")
    st.exception(error)
    st.stop()


# ==================================================
# 4. Upload de l'image
# ==================================================

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)


# ==================================================
# 5. Prédiction
# ==================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Image IRM sélectionnée",
            use_container_width=True
        )

    except Exception as error:
        st.error("Impossible de lire l'image.")
        st.exception(error)
        st.stop()

    if st.button(
        "Analyser l'image",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner("Analyse en cours..."):

                # Sauvegarder temporairement l'image
                temp_path = (
                    BASE_DIR
                    / f"temp_image{Path(uploaded_file.name).suffix.lower()}"
                )

                temp_path.write_bytes(uploaded_file.getvalue())

                try:
                    # Prédiction avec la fonction du projet
                    predicted_class, confidence, probabilities = (
                        predict_image(model, temp_path, preprocess_input, label_encoder)
                    )

                finally:
                    # Supprimer le fichier temporaire
                    if temp_path.exists():
                        temp_path.unlink()

            # Convertir les probabilités en tableau NumPy
            probabilities = np.asarray(
                probabilities,
                dtype=np.float32
            ).reshape(-1)

            # Vérifier la compatibilité avec le LabelEncoder
            if len(probabilities) != len(label_encoder.classes_):
                raise ValueError(
                    "Le nombre de probabilités ne correspond pas "
                    "au nombre de classes du LabelEncoder. "
                    f"Probabilités : {len(probabilities)}, "
                    f"Classes : {len(label_encoder.classes_)}."
                )

            if not np.all(np.isfinite(probabilities)):
                raise ValueError(
                    "Les probabilités contiennent des valeurs invalides."
                )

            # Résultat principal
            st.success(f"Classe prédite : {predicted_class}")

            st.metric(
                label="Confiance de la prédiction",
                value=f"{float(confidence):.2%}"
            )

            # Probabilités par classe
            st.subheader("Probabilités par classe")

            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities
            ):
                probability = float(probability)

                st.write(
                    f"**{class_name}** : {probability:.2%}"
                )

                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

        except Exception as error:
            st.error(
                "Une erreur est survenue pendant la prédiction."
            )

            # Afficher le détail de l'erreur pour le débogage
            st.exception(error)

            print(traceback.format_exc())


# ==================================================
# 6. Avertissement médical
# ==================================================

st.divider()

st.caption(
    "Outil expérimental à but éducatif. "
    "Les prédictions ne constituent pas un diagnostic médical."
)
