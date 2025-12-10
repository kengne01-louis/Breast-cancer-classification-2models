import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# GESTION DES DÉPENDANCES 
import sys
import subprocess

def check_dependencies():
    """Vérifie et installe les dépendances manquantes"""
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        import streamlit as st
        print("✅ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"⚠️ Dépendance manquante: {e}")
        print("🔄 Tentative d'installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dépendances installées avec succès")
            return True
        except Exception as install_error:
            print(f"❌ Échec d'installation: {install_error}")
            return False

# Exécuter la vérification
if __name__ == "__main__":
    if not check_dependencies():
        print("❌ Impossible de résoudre les dépendances")
        sys.exit(1)


# Configuration de la page
st.set_page_config(
    page_title="🌐 CLASSIFICATION DU CANCER DU SEIN ",
    page_icon="🌐",
    layout="centered",
)

# fond bleu professionnel
st.markdown("""
<style>
    /* Fond bleu professionnel */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
        background-attachment: fixed;
    }
    
    /* Style pour les cartes */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15);
        border: 2px solid #2196f3;
    }
    
    /* Style pour les titres */
    .custom-title {
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Style pour les sous-titres */
    .custom-subtitle {
        color: #0d47a1;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Uploader stylisé */
    .stFileUploader > div > div {
        border: 3px dashed #1565c0;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #0d47a1;
        background: rgba(255, 255, 255, 1);
    }
    
    /* Style pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e3f2fd;
        border-radius: 10px 10px 0px 0px;
        gap: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2196f3 !important;
        color: white !important;
    }
    
    /* Style pour la sélection de modèle */
    .model-selection-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 25px rgba(33, 150, 243, 0.2);
        border: 3px solid #2196f3;
    }
    
    .model-button {
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        border: 2px solid #2196f3;
    }
    
    .model-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .model-selected {
        background: linear-gradient(135deg, #2196f3 0%, #1565c0 100%);
        color: white;
        border: 2px solid #0d47a1;
    }
    
    .model-not-selected {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    /* Footer personnalisé */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: white;
        padding: 10px 0;
        text-align: center;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .footer-content {
        display: flex;
        justify-content: space-around;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    
    .footer-icon {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Ajout du footer avec vos coordonnées
st.markdown("""
<div class="custom-footer">
    <div class="footer-content">
        <div class="footer-item">
            <span class="footer-icon">📞</span>
            <span><strong>Téléphone :</strong> +237 659 06 06 81</span>
        </div>
        <div class="footer-item">
            <span class="footer-icon">📧</span>
            <span><strong>Email :</strong> louiskngn01@gmail.com</span>
        </div>
        <div class="footer-item">
            <span class="footer-icon">©</span>
            <span>Yaoundé-Cameroun:2025</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

#  NOUVELLE SECTION : SÉLECTION DU MODÈLE 
st.markdown("""
    <div class="custom-title">🌐 CLASSIFICATION DU CANCER DU SEIN</div>
    <div class="custom-subtitle">Sélectionnez d'abord le modèle, puis importez votre image</div>
""", unsafe_allow_html=True)

# Section de sélection du modèle
st.markdown('<div class="model-selection-card">', unsafe_allow_html=True)
st.markdown("### 💎**SÉLECTION DU MODÈLE D'ANALYSE**💎")

# Créer deux colonnes pour les options de modèle
col1, col2 = st.columns(2)

# Initialiser la variable de session pour le modèle si elle n'existe pas
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

with col1:
    if st.button("**Modèle 1 : CNN**", use_container_width=True, 
                 type="primary" if st.session_state.selected_model == "CNN" else "secondary"):
        st.session_state.selected_model = "CNN"

with col2:
    if st.button("**Modèle 2 : TRANSFER LEARNING**", use_container_width=True,
                 type="primary" if st.session_state.selected_model == "Transfer Learning" else "secondary"):
        st.session_state.selected_model = "Transfer Learning"

# Afficher le modèle sélectionné
if st.session_state.selected_model:
    st.markdown(f"""
    <div style="text-align: center; padding: 15px; margin-top: 20px; 
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                border-radius: 10px; border: 2px solid #2196f3;">
        <h4 style="color: #1565c0;">✔ MODÈLE SÉLECTIONNÉ : <strong>{st.session_state.selected_model}</strong></h4>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 15px; margin-top: 20px; 
                background: #fff3e0; border-radius: 10px; border: 2px solid #ff9800;">
        <h4 style="color: #e65100;">⚠️ Veuillez sélectionner un modèle</h4>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# CHARGEMENT DES MODÈLES 
@st.cache_resource
def load_cnn_model():
    MODEL_PATH = "models/mon_CNN_final.h5"
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_transfer_learning_model():
    """Charger le modèle Transfer Learning"""
    try:
        TRANSFER_MODEL_PATH = "models/efficientnet_final_model.h5"
        return tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.info("Veuillez vérifier que le fichier 'efficientnet_final_model.h5' existe.")
        st.stop()

# Vérifier si un modèle est sélectionné avant de continuer
if not st.session_state.selected_model:
    st.warning("🔹Veuillez d'abord sélectionner un modèle ci-dessus avant de continuer.")
else:
    # Charger le modèle selon la sélection
    if st.session_state.selected_model == "CNN":
        model = load_cnn_model()
        model_name = "CNN"
        input_size = (64, 64)  # Taille pour le modèle CNN
    else:
        model = load_transfer_learning_model()
        model_name = "Transfer Learning"
        input_size = (224, 224)  # Taille pour EfficientNetB0

    # CLASSES 
    class_names = ["[0]=malignant", "[1]=normal"]

    # PRÉTRAITEMENT DYNAMIQUE selon le modèle
    def preprocess_image(image, model_type=model_name):
        """Prétraitement adapté à chaque modèle"""
        if model_type == "CNN":
            # Pour CNN : 64x64
            image = image.resize((64, 64))
        else:
            # Pour Transfer Learning (EfficientNetB0) : 224x224
            image = image.resize((224, 224))
        
        image = np.array(image) / 255.0
        
        if image.shape[-1] == 4:  # RGBA -> RGB
            image = image[..., :3]
        
        image = np.expand_dims(image, 0)
        return image

    # PREDICTION 
    def predict_image(image):
        processed = preprocess_image(image, model_name)
        pred = model.predict(processed, verbose=0)[0]

        if pred.shape == () or len(pred) == 1:
            prob1 = float(pred)
            prob0 = 1 - prob1
            probs = [prob0, prob1]
        else:
            probs = pred.tolist()

        predicted_index = int(np.argmax(probs))
        predicted_class = class_names[predicted_index]
        confidence = probs[predicted_index] * 100

        # Ajouter le nom du modèle dans les résultats
        predicted_class = f"{predicted_class} (Modèle: {model_name})"
        
        return predicted_class, confidence, probs, predicted_index

    # Fonction principale pour afficher les résultats
    def display_results(image):
        """Affiche les résultats de l'analyse"""
        # Section image
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.image(image, caption=f"✅ Image analysée avec {model_name} ✅", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyse
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.write(f" **ANALYSE EN COURS AVEC {model_name.upper()}...**")
        progress_bar = st.progress(0)
        for i in range(101):
            progress_bar.progress(i)
        st.write("✅ Analyse terminée !")
        st.markdown('</div>', unsafe_allow_html=True)

        # PREDICTION
        predicted_class, confidence, probs, predicted_index = predict_image(image)

        # RÉSULTAT
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        if predicted_index == 0:  # malignant
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                         border-radius: 10px; padding: 20px; border-left: 6px solid #f44336;">
                    <h2 style="color:#d32f2f;">⚠️ RÉSULTAT : {predicted_class}</h2>
                    <h3 style="color:#b71c1c;">Confiance : {confidence:.2f}%</h3>
                    <p style="color:#d32f2f;"><strong>Modèle utilisé : {model_name}</strong></p>
                    <p style="color:#d32f2f;"><small>Taille d'entrée : {input_size[0]}x{input_size[1]}</small></p>
                </div>
            """, unsafe_allow_html=True)
        else:  # normal
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                         border-radius: 10px; padding: 20px; border-left: 6px solid #4caf50;">
                    <h2 style="color:#2e7d32;">✅ RÉSULTAT : {predicted_class}</h2>
                    <h3 style="color:#1b5e20;">Confiance : {confidence:.2f}%</h3>
                    <p style="color:#2e7d32;"><strong>Modèle utilisé : {model_name}</strong></p>
                    <p style="color:#2e7d32;"><small>Taille d'entrée : {input_size[0]}x{input_size[1]}</small></p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # PROBABILITÉS
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"### 🔻**PROBABILITÉS DÉTAILLÉES ({model_name})**🔻")
        
        for i, cls in enumerate(class_names):
            if i < len(probs):
                bar_color = "#f44336" if i == 0 else "#4caf50"
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{cls}**")
                with col2:
                    st.write(f"**{probs[i]*100:.2f}%**")
                
                st.markdown(f"""
                    <div style="background: #f5f5f5; border-radius: 10px; height: 25px; margin: 5px 0;">
                        <div style="background: {bar_color}; width: {probs[i]*100}%; 
                                 height: 100%; border-radius: 10px;"></div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # CONCLUSION
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        if predicted_index == 0:
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #ffebee; 
                         border-radius: 10px; border: 2px solid #f44336;">
                    <h2 style="color:#d32f2f;">⚠️⚠️ CONCLUSION ({model_name})</h2>
                    <h3 style="color:#b71c1c;">Votre sein est Cancereux.</h3>
                    <p style="color:#d32f2f;">Analyse réalisée avec le modèle {model_name}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #e8f5e9; 
                         border-radius: 10px; border: 2px solid #4caf50;">
                    <h2 style="color:#2e7d32;">✅ CONCLUSION ({model_name})</h2>
                    <h3 style="color:#1b5e20;">Votre sein est en forme normal.</h3>
                    <p style="color:#2e7d32;">Analyse réalisée avec le modèle {model_name}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # MESSAGE FINAL
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        if predicted_index == 0:
            st.markdown(f"""
                <div style="padding: 15px; background: #fff3e0; border-radius: 10px;">
                    <p style="color:#e65100;">
                        ⚠️⚠️ <b>IMPORTANT :</b> Consultez un professionnel de santé rapidement.
                    </p>
                    <p style="color:#e65100;">
                        <i>Résultat obtenu avec le modèle {model_name}</i>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="padding: 15px; background: #f1f8e9; border-radius: 10px;">
                    <p style="color:#33691e;">
                        ✅ <b>INFORMATION :</b> Continuez vos examens de routine.
                    </p>
                    <p style="color:#33691e;">
                        <i>Résultat obtenu avec le modèle {model_name}</i>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ONGLETS POUR CHOIX DE MÉTHODE ==========

    # Créer des onglets pour choisir la méthode d'importation
    tab1, tab2 = st.tabs(["📁 **Importer une image**", "📸 **Prendre une photo**"])

    # Onglet 1 : Importer une image
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"### 📁 **IMPORTER UNE IMAGE (Modèle : {model_name})**")
        uploaded_file = st.file_uploader("**Sélectionnez une image**", 
                                        type=["jpg", "jpeg", "png"],
                                        help="Format accepté : JPG, JPEG, PNG")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            display_results(image)
        else:
            st.markdown(f"""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #1565c0;">📁 Sélectionnez une image depuis votre appareil</h3>
                    <p style="color: #666;">Format accepté : JPG, JPEG, PNG</p>
                    <p style="color: #2196f3;"><strong>Modèle sélectionné : {model_name}</strong></p>
                    <p style="color: #2196f3;"><small>Taille d'entrée : {input_size[0]}x{input_size[1]} pixels</small></p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Onglet 2 : Prendre une photo
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"### 📸 **PRENDRE UNE PHOTO (Modèle : {model_name})**")
        st.markdown(f"""
            <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <p style="color: #1565c0;">
                    <b>Instructions :</b><br>
                    1. Autorisez l'accès à votre caméra<br>
                    2. Placez-vous face à la caméra<br>
                    3. Cliquez sur le bouton pour prendre la photo<br>
                    4. L'analyse avec <strong>{model_name}</strong> commencera automatiquement<br>
                    5. <strong>Taille d'entrée : {input_size[0]}x{input_size[1]} pixels</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour activer la caméra
        use_camera = st.checkbox("✅ Activer la caméra", value=False)
        
        if use_camera:
            st.info(f"🎥 La caméra est activée. Analyse avec {model_name} (taille: {input_size[0]}x{input_size[1]}).")
            camera_image = st.camera_input("**Prendre une photo**", 
                                          key="camera_capture",
                                          help="Cliquez sur le bouton de l'appareil photo pour capturer l'image")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                display_results(image)
            else:
                st.markdown(f"""
                    <div style="text-align: center; padding: 40px;">
                        <h3 style="color: #1565c0;">📸 Préparez-vous à prendre une photo</h3>
                        <p style="color: #666;">Positionnez-vous face à la caméra et cliquez sur le bouton de capture</p>
                        <p style="color: #2196f3;"><strong>Modèle : {model_name}</strong></p>
                        <p style="color: #2196f3;"><small>Taille d'entrée : {input_size[0]}x{input_size[1]} pixels</small></p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #1565c0;">📸 Activez la caméra pour prendre une photo</h3>
                    <p style="color: #666;">Cochez la case ci-dessus pour activer votre caméra</p>
                    <p style="color: #2196f3;"><strong>Modèle : {model_name}</strong></p>
                    <p style="color: #2196f3;"><small>Taille d'entrée : {input_size[0]}x{input_size[1]} pixels</small></p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Message d'information générale
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h4 style="color: #1565c0;">🔻INFORMATIONS🔻</h4>
            <p><strong>Modèle 1 (CNN) :</strong> Réseaux de Neurones Convolutif - Taille d'entrée : 64x64 pixels</p>
            <p><strong>Modèle 2 (TRANSFER LEARNING) :</strong> EfficientNetB0 - Taille d'entrée : 224x224 pixels</p>
            <p style="color: #0d47a1; font-weight: bold;">Modèle actuellement sélectionné : {model_name}</p>
            <p style="color: #0d47a1;">Taille d'entrée : {input_size[0]}x{input_size[1]} pixels</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)