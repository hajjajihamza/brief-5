import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from src.model import predict_image


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "meilleur_modele.keras"
ENCODER_PATH = BASE_DIR.parent / "label_encoder.pkl"

# Configuration de l'application
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NeuroVision AI")
st.subheader("Classification des images IRM cérébrales")

st.write(
    "Téléchargez une image IRM pour tester "
    "la prédiction du modèle CNN."
)

# Chargement du modèle et du LabelEncoder
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

except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Erreur de chargement : {error}")
    st.stop()

# Interface : upload de l'image

uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)

# Affichage et prédiction

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image IRM sélectionnée",
        use_container_width=True
    )

    if st.button(
        "Analyser l'image",
        type="primary",
        use_container_width=True
    ):
        temp_path = (
            BASE_DIR
            / f"temp_image{Path(uploaded_file.name).suffix.lower()}"
        )

        try:
            # Sauvegarder temporairement l'image
            temp_path.write_bytes(
                uploaded_file.getvalue()
            )

            # Prédire
            with st.spinner("Analyse en cours..."):
                predicted_class, confidence, probabilities = (
                    predict_image(temp_path)
                )

            # Résultat principal
            st.success(
                f"Classe prédite : {predicted_class}"
            )

            st.metric(
                label="Probabilité associée à la classe prédite",
                value=f"{confidence:.2%}"
            )

            # Probabilités de toutes les classes
            st.subheader("Probabilités par classe")

            results = {
                class_name: float(probability)
                for class_name, probability in zip(
                    label_encoder.classes_,
                    probabilities
                )
            }

            for class_name, probability in results.items():
                st.write(
                    f"**{class_name}** : {probability:.2%}"
                )

                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

        except (ValueError, FileNotFoundError, OSError) as error:
            st.error(str(error))

        except Exception:
            st.error(
                "Une erreur inattendue est survenue "
                "pendant la prédiction."
            )

        finally:
            # Supprimer le fichier temporaire
            if temp_path.exists():
                temp_path.unlink()


# Avertissement
st.divider()

st.caption(
    "Outil expérimental à but éducatif. "
    "Les prédictions ne constituent pas un diagnostic médical."
)