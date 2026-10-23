from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, BackupAndRestore, EarlyStopping
import mlflow
from .dataset import create_dataset
from .const import SIZE_IMAGE, BATCH_SIZE, EPOCHS
from .visualization import afficher_courbes_apprentissage

def evaluate_model(model, test_ds, model_path ,y_test, name, label_encoder):
    # Predire les probabilites sur le dataset de test
    y_pred_probs = model.predict(test_ds, verbose=0) # Probabiliter sur les 4 classes
    # print(y_pred_probs[0])

    # Recuperer la classe avec la probabilite maximale
    y_pred = np.argmax(y_pred_probs, axis=1) # index de max proba sur les column 

    # Convertir les labels one-hot en numeros de classes
    y_true = np.argmax(y_test, axis=1) # convert One hot encoding à label encoded

    # Evaluer sur le dataset de test
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    f1 = f1_score(y_true, y_pred, average='macro')

    # Enregistrer les metriques dans MLflow
    mlflow.log_metric("test_loss", float(test_loss))
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("test_f1_score", float(f1))

    print(
        f"Resultats : Loss={test_loss:.4f} | "
        f"Accuracy={test_accuracy:.4f} | "
        f"F1-score={f1:.4f}"
    )

    # ================= MATRICE DE CONFUSION =================
    classes = label_encoder.classes_

    cm = confusion_matrix(
        y_true, 
        y_pred,
        labels=np.arange(len(classes))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    # Sauvegarder l'image
    cm_path = f"{model_path}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")

    # Enregistrer dans MLflow comme artifact
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    # ================= RAPPORT DE CLASSIFICATION =================
    # Generer le rapport
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )

    print(report)

    # Sauvegarder le rapport dans un fichier TXT
    report_path = f"{model_path}/classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # Enregistrer dans MLflow comme artifact
    mlflow.log_artifact(
        report_path,
        artifact_path="evaluation"
    )

    # ================= ENREGISTREMENT MLFLOW =================
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        registered_model_name=name
    )


def build_transfer_model(base_model, normalize_collback, x_train, y_train, x_val, y_val, x_test,y_test,data_augmentation):
    # Charger le modele pre-entraine
    base = base_model(
        weights='imagenet', # loader les weights of train imagenet
        include_top=False, # remove classes of predicts base
        input_shape=(SIZE_IMAGE, SIZE_IMAGE, 3) # Size of input
    )

    # ================= DATASETS =================
    # Trian
    train_ds = create_dataset(x_train, y_train, normalize_collback= normalize_collback, is_use_shuffle=True)

    # Validation 
    val_ds = create_dataset(x_val, y_val, normalize_collback)

    # Test
    test_ds = create_dataset(x_test, y_test, normalize_collback)

    # ================= DOSSIER =================
    model_path = f"models/{base.name}"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # ================= TRANSFER LEARNING (Inference) =================
    
    base.trainable = False # franze all parametres of base model to not chnager on training

    # Architecture Sequential
    model = Sequential([
        base, # Extraire les caractéristiques en mode inference

        # Appliquer augmentation
        data_augmentation,

        GlobalAveragePooling2D(), # minimizer les Featurs Map à vectore 

        # hiden layer
        Dense(128, activation="relu"),
        Dropout(0.3),

        # Output layer
        Dense(4, activation="softmax")
    ])

    model.summary()

    # Compilation du modele
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Optimizer avec learninig rate
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks du Transfer Learning
    callbacks_tl = [
        # Suvgarder le baset model pendant le training
        ModelCheckpoint(
            filepath=f"{model_path}/best_model_tl.keras", # path de bast model
            monitor="val_loss", # monitore de comparisent
            save_best_only=True, # Sauvgarder le baset model sulment
            mode="min", # Suavgarder le model de min monitor
            verbose=1 # afficher sur terminal
        ),
        # Sauvgarder le étap de train si ariter pour pas perdet le étap de training
        BackupAndRestore(
            backup_dir=f"{model_path}/backup_tl" # dosser de sauvgarder
        ),
        # Limiter le surapprentissage
        EarlyStopping(
            monitor="val_loss", # monitore de comparisent
            patience=5, # nomber des epochs sans augmentation de monitor
            restore_best_weights=True, # returner baset weights
            mode="min", # minimuent monitor
            verbose=1
        )
    ]

    # ================= RUN 1 : TRANSFER LEARNING =================

    with mlflow.start_run(run_name=f"{base.name}_feature_extraction"):
        
        mlflow.set_tag("training_stage", "transfer_learning")

        mlflow.log_params({
            "base_model": base.name,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "image_size": SIZE_IMAGE,
            "learning_rate": 0.001,
        })

        print('\n========= Tresfer Learning =========\n')

        # Entrainement du Transfer Learning
        start_time = time.time()

        trensfoer_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks_tl
        )

        train_time = time.time() - start_time

        # Enregistrer les mtrics
        mlflow.log_metric("training_duration_TL_seconds", train_time)

        print(f"Temps d'entraînement en Trensfor learning : {train_time / 60:.2f} minute")

        # ================= EVALUATION =================
        evaluate_model(model, test_ds, model_path, y_test, base.name)

        # ================= COURBES =================
        afficher_courbes_apprentissage(trensfoer_history.history)

    # ================= FINE-TUNING (Training) =================
    
    # Autoriser l'entrainement des couches du modele de base
    base.trainable = True

    # Frazer tout les layers de 0 jusqua dérnier 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Rester les 30 derniéres couches au mode inference si est instence à BatchNormalization
    # Garder BatchNormalization non entrainable
    for layer in base.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    # Recompiler apres la modification de trainable
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Optimizer avec learninig rate tres petit
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks du Fine-Tuning
    callbacks_ft = [
        ModelCheckpoint(
            filepath=f"{model_path}/best_model_ft.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        BackupAndRestore(
            backup_dir=f"{model_path}/backup_ft"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        )
    ]

    # ================= RUN 2 : FINE-TUNING =================
    with mlflow.start_run(run_name=f"{base.name}_fine_tuning"):

        mlflow.set_tag("training_stage", "fine_tuning")

        mlflow.log_params({
            "base_model": base.name,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "image_size": SIZE_IMAGE,
            "learning_rate": 0.00001,
            "unfrozen_last_layers": 30,
            "batch_normalization_frozen": True
        })

        print('\n========= Fine Tuning =========\n')

        # Entrainement du Fine-Tuning
        start_time = time.time()

        fine_tuning_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks_ft  
        )

        train_time = time.time() - start_time

        # Enregistrer les mtrics
        mlflow.log_metric("training_duration_FT_seconds", train_time)
        
        print(f"Temps d'entraînement en Fine tuning : {train_time / 60:.2f} minute")

        # ================= EVALUATION =================
        evaluate_model(model, test_ds, model_path, y_test, base.name)

        # ================= COURBES =================
        afficher_courbes_apprentissage(fine_tuning_history.history)
