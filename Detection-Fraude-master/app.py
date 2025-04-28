import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.callbacks import Callback
# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude Bancaire | IA Analytics",
    layout="wide",
    page_icon="💳",
    initial_sidebar_state="expanded"
)

# Style personnalisé avec un thème moderne
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #3366ff, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0099cc;
        margin-top: 2rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3366ff;
    }
    .stButton>button {
        background-color: #3366ff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #254EDB;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .alert-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .comparison-table {
        width: 100%;
        text-align: center;
    }
    .comparison-table th {
        background-color: #f0f8ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f8ff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3366ff !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Fonctions -----------

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown('<div class="alert-success">✅ Données chargées avec succès!</div>', unsafe_allow_html=True)
            return data
        except Exception as e:
            st.markdown(f'<div class="alert-warning">⚠️ Erreur de chargement: {e}</div>', unsafe_allow_html=True)
    return None

def show_data_info(data):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Aperçu des données</div>', unsafe_allow_html=True)
        st.dataframe(data.head(), use_container_width=True)
        
        st.markdown('<div class="sub-header">Statistiques descriptives</div>', unsafe_allow_html=True)
        st.dataframe(data.describe(), use_container_width=True)
        
    with col2:
        st.markdown('<div class="sub-header">Distribution de la variable "Amount"</div>', unsafe_allow_html=True)
        fig = px.histogram(data, x="Amount", marginal="box", nbins=50, 
                          color_discrete_sequence=['#3366ff'],
                          title="Distribution des montants de transaction")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="sub-header">Distribution des classes (fraude vs non-fraude)</div>', unsafe_allow_html=True)
        class_counts = data['Class'].value_counts().reset_index()
        class_counts.columns = ['Classe', 'Nombre']
        class_counts['Classe'] = class_counts['Classe'].map({0: 'Légitime', 1: 'Fraude'})
        
        fig = px.pie(class_counts, values='Nombre', names='Classe', 
                    color_discrete_sequence=['#3366ff', '#ff3366'],
                    title="Répartition des transactions")
        st.plotly_chart(fig, use_container_width=True)

def preprocess_data(data):
    st.markdown('<div class="alert-info">🔄 Prétraitement des données en cours...</div>', unsafe_allow_html=True)
    
    # Équilibrage des données (optionnel)
    if st.session_state.get('balance_data', False):
        fraud_samples = data[data['Class'] == 1]
        non_fraud_samples = data[data['Class'] == 0].sample(n=len(fraud_samples)*3, random_state=42)
        data = pd.concat([fraud_samples, non_fraud_samples])
        st.markdown(f'<div class="alert-info">ℹ️ Données rééquilibrées: {len(fraud_samples)} fraudes, {len(non_fraud_samples)} non-fraudes</div>', unsafe_allow_html=True)
    
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=st.session_state.get('test_size', 0.2), 
        stratify=y, random_state=42
    )
    
    # Mise à l'échelle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test

def train_mlp(X_train, y_train):
    neurons = st.session_state.get('mlp_neurons', [32, 16])
    dropout_rate = st.session_state.get('dropout_rate', 0.2)
    epochs = st.session_state.get('epochs', 10)
    
    model = Sequential()
    model.add(Dense(neurons[0], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    
    for n in neurons[1:]:
        model.add(Dense(n, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Ajout d'une barre de progression personnalisée
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class TrainingCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Entraînement en cours... Epoch {epoch+1}/{epochs}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping, TrainingCallback()],
        verbose=0
    )
    
    progress_bar.progress(1.0)
    status_text.text("Entraînement terminé!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return model, history

def train_autoencoder(X_train):
    encoding_dim = st.session_state.get('encoding_dim', [16, 8])
    epochs = st.session_state.get('epochs', 10)
    
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = input_layer
    for dim in encoding_dim:
        encoded = Dense(dim, activation='relu')(encoded)
    
    # Decoder
    decoded = encoded
    for dim in reversed(encoding_dim[:-1]):  # Ne pas répéter la dernière dimension
        decoded = Dense(dim, activation='relu')(decoded)
    
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Ajout d'une barre de progression personnalisée
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class TrainingCallback(Callback):

        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Entraînement en cours... Epoch {epoch+1}/{epochs}")
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[TrainingCallback()],
        verbose=0
    )
    
    progress_bar.progress(1.0)
    status_text.text("Entraînement terminé!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return autoencoder, history

def get_autoencoder_predictions(model, X_test, y_test):
    X_test_pred = model.predict(X_test)
    reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)
    
    # Sélection automatique du seuil ou spécifié par l'utilisateur
    if st.session_state.get('custom_threshold'):
        threshold = st.session_state.get('threshold_value', 0.95)
    else:
        # Trouver le meilleur seuil basé sur F1-score
        thresholds = np.linspace(np.min(reconstruction_error), np.max(reconstruction_error), 100)
        best_f1 = 0
        best_threshold = np.percentile(reconstruction_error, 95)
        
        for t in thresholds:
            preds = (reconstruction_error > t).astype(int)
            tp = np.sum((preds == 1) & (y_test == 1))
            fp = np.sum((preds == 1) & (y_test == 0))
            fn = np.sum((preds == 0) & (y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        threshold = best_threshold
    
    preds_binary = (reconstruction_error > threshold).astype(int)
    return preds_binary, reconstruction_error, threshold

def display_results(y_true, y_pred, y_score, model_type, X_test_raw=None, threshold=None):
    st.markdown(f'<div class="sub-header">Résultats du modèle {model_type}</div>', unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    # Calcul des métriques
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Affichage des métriques dans des cards
    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Précision</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        '''.format(precision*100), unsafe_allow_html=True)
        
    with col2:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Rappel</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        '''.format(recall*100), unsafe_allow_html=True)
        
    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Score F1</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        '''.format(f1*100), unsafe_allow_html=True)
        
    with col4:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-title">Exactitude</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        '''.format(accuracy*100), unsafe_allow_html=True)
    
    # Onglets pour les différentes visualisations
    tabs = st.tabs(["Matrice de confusion", "Courbes ROC/PR", "Détails"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cm = confusion_matrix(y_true, y_pred)
            fig = px.imshow(cm, 
                          labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                          x=["Non-fraude", "Fraude"],
                          y=["Non-fraude", "Fraude"],
                          text_auto=True,
                          color_continuous_scale="Blues")
            fig.update_layout(title="Matrice de confusion")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Interprétation de la matrice")
            st.markdown(f"""
            - **Vrais négatifs (TN)**: {tn} transactions légitimes correctement identifiées
            - **Faux positifs (FP)**: {fp} transactions légitimes incorrectement classées comme fraudes
            - **Faux négatifs (FN)**: {fn} fraudes manquées (classées comme légitimes)
            - **Vrais positifs (TP)**: {tp} fraudes correctement détectées
            
            Le modèle a un taux de détection de fraude de **{recall*100:.1f}%** et un taux de faux positifs de **{fp/(fp+tn)*100:.1f}%**.
            """)
            
            if model_type == "Autoencodeur" and threshold is not None:
                st.markdown(f"**Seuil d'anomalie utilisé**: {threshold:.6f}")
    
    with tabs[1]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(dash='dash', color='grey')))
            fig.update_layout(
                title='Courbe ROC',
                xaxis=dict(title='Taux de faux positifs (1 - Spécificité)'),
                yaxis=dict(title='Taux de vrais positifs (Sensibilité)'),
                legend=dict(x=0.01, y=0.99),
                width=400, height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
            ap_score = average_precision_score(y_true, y_score)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', 
                                    name=f'PR (AP = {ap_score:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[sum(y_true)/len(y_true)]*2, mode='lines', 
                                    name='Aléatoire', line=dict(dash='dash', color='grey')))
            fig.update_layout(
                title='Courbe Précision-Rappel',
                xaxis=dict(title='Rappel'),
                yaxis=dict(title='Précision'),
                legend=dict(x=0.01, y=0.99),
                width=400, height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Rapport de classification détaillé")
        st.text(classification_report(y_true, y_pred))
        
        if X_test_raw is not None:
            st.markdown("### Échantillon de prédictions")
            result_df = pd.DataFrame(X_test_raw).copy()
            result_df['Classe_Réelle'] = y_true
            result_df['Prédiction'] = y_pred
            result_df['Score'] = y_score
            
            # Filtrer pour montrer des exemples intéressants
            st.markdown("#### Fraudes détectées (Vrais Positifs)")
            true_positives = result_df[(result_df['Classe_Réelle'] == 1) & (result_df['Prédiction'] == 1)].head(5)
            st.dataframe(true_positives, use_container_width=True)
            
            st.markdown("#### Fraudes manquées (Faux Négatifs)")
            false_negatives = result_df[(result_df['Classe_Réelle'] == 1) & (result_df['Prédiction'] == 0)].head(5)
            st.dataframe(false_negatives, use_container_width=True)
    
    # Stocker les résultats dans la session pour comparaison
    if model_type == "MLP":
        st.session_state['mlp_results'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc_score if 'auc_score' in locals() else None,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score
        }
    elif model_type == "Autoencodeur":
        st.session_state['autoencoder_results'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc_score if 'auc_score' in locals() else None,
            'threshold': threshold,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_score': y_score
        }

def compare_models():
    if 'mlp_results' not in st.session_state or 'autoencoder_results' not in st.session_state:
        st.markdown('<div class="alert-warning">⚠️ Veuillez entraîner les deux modèles pour les comparer</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="sub-header">Comparaison des performances des modèles</div>', unsafe_allow_html=True)
    
    mlp = st.session_state['mlp_results']
    ae = st.session_state['autoencoder_results']
    
    # Tableau comparatif
    comparison_data = {
        'Métrique': ['Précision', 'Rappel', 'Score F1', 'Exactitude', 'AUC'],
        'MLP': [
            f"{mlp['precision']*100:.1f}%", 
            f"{mlp['recall']*100:.1f}%", 
            f"{mlp['f1']*100:.1f}%", 
            f"{mlp['accuracy']*100:.1f}%",
            f"{mlp['auc']:.3f}" if mlp['auc'] is not None else "N/A"
        ],
        'Autoencodeur': [
            f"{ae['precision']*100:.1f}%", 
            f"{ae['recall']*100:.1f}%", 
            f"{ae['f1']*100:.1f}%", 
            f"{ae['accuracy']*100:.1f}%",
            f"{ae['auc']:.3f}" if ae['auc'] is not None else "N/A"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Graphique comparatif
    metrics = ['Précision', 'Rappel', 'F1', 'Exactitude']
    mlp_values = [mlp['precision'], mlp['recall'], mlp['f1'], mlp['accuracy']]
    ae_values = [ae['precision'], ae['recall'], ae['f1'], ae['accuracy']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=[v*100 for v in mlp_values],
        name='MLP',
        marker_color='#3366ff'
    ))
    fig.add_trace(go.Bar(
        x=metrics,
        y=[v*100 for v in ae_values],
        name='Autoencodeur',
        marker_color='#ff3366'
    ))
    
    fig.update_layout(
        title='Comparaison des métriques de performance',
        xaxis=dict(title='Métrique'),
        yaxis=dict(title='Valeur (%)'),
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des cas divergents
    st.markdown("### Analyse des cas divergents")
    st.markdown("""
    Cette section montre les transactions où les deux modèles sont en désaccord.
    Ces cas peuvent révéler les forces et faiblesses de chaque approche.
    """)
    
    mlp_pred = mlp['y_pred']
    ae_pred = ae['y_pred']
    y_true = mlp['y_true']
    
    # Créer un DataFrame pour l'analyse
    divergent_df = pd.DataFrame({
        'Classe_Réelle': y_true,
        'Prédiction_MLP': mlp_pred,
        'Prédiction_AE': ae_pred,
        'Score_MLP': mlp['y_score'],
        'Score_AE': ae['y_score']
    })
    
    # Filtrer les cas divergents
    divergent_cases = divergent_df[divergent_df['Prédiction_MLP'] != divergent_df['Prédiction_AE']]
    
    if len(divergent_cases) > 0:
        st.markdown(f"**{len(divergent_cases)} transactions** ont été classées différemment par les deux modèles.")
        
        # Analyse détaillée des cas divergents
        divergent_stats = pd.DataFrame({
            'Scénario': [
                'MLP: Fraude, AE: Non-fraude, Réalité: Fraude',
                'MLP: Fraude, AE: Non-fraude, Réalité: Non-fraude',
                'MLP: Non-fraude, AE: Fraude, Réalité: Fraude',
                'MLP: Non-fraude, AE: Fraude, Réalité: Non-fraude'
            ],
            'Nombre': [
                len(divergent_cases[(divergent_cases['Prédiction_MLP'] == 1) & 
                                    (divergent_cases['Prédiction_AE'] == 0) & 
                                    (divergent_cases['Classe_Réelle'] == 1)]),
                len(divergent_cases[(divergent_cases['Prédiction_MLP'] == 1) & 
                                    (divergent_cases['Prédiction_AE'] == 0) & 
                                    (divergent_cases['Classe_Réelle'] == 0)]),
                len(divergent_cases[(divergent_cases['Prédiction_MLP'] == 0) & 
                                    (divergent_cases['Prédiction_AE'] == 1) & 
                                    (divergent_cases['Classe_Réelle'] == 1)]),
                len(divergent_cases[(divergent_cases['Prédiction_MLP'] == 0) & 
                                    (divergent_cases['Prédiction_AE'] == 1) & 
                                    (divergent_cases['Classe_Réelle'] == 0)])
            ]
        })
        
        st.table(divergent_stats)
        
        # Échantillon de cas divergents
        st.dataframe(divergent_cases.head(10), use_container_width=True)
        
        # Recommandation d'ensemble
        st.markdown("### Recommandation pour un système de détection combiné")
        st.markdown("""
        Basé sur l'analyse des cas divergents, nous recommandons:
        
        1. **Utiliser les deux modèles en parallèle**
        2. **Pour les transactions critiques/de montant élevé**: Considérer comme suspecte toute transaction signalée par l'un des deux modèles (Union des prédictions)
        3. **Pour les transactions standard**: Ne considérer comme suspecte que les transactions signalées par les deux modèles (Intersection des prédictions)
        """)
    else:
        st.info("Les deux modèles sont en accord sur toutes les prédictions.")

# ----------- Interface principale -----------

def main():
    # Initialisation de la session state
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['test_size'] = 0.2
        st.session_state['mlp_neurons'] = [32, 16]
        st.session_state['dropout_rate'] = 0.2
        st.session_state['encoding_dim'] = [16, 8]
        st.session_state['epochs'] = 10
        st.session_state['balance_data'] = False
        st.session_state['custom_threshold'] = False
        st.session_state['threshold_value'] = 0.95
    
    # En-tête
    st.markdown('<h1 class="main-header">💳 Détection de Fraude Bancaire par Deep Learning</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        Une approche KDD (Knowledge Discovery in Database) avec Tensorflow/Keras pour détecter les transactions frauduleuses
    </p>
    """, unsafe_allow_html=True)

    # Sidebar pour les options et chargement
    with st.sidebar:
        st.markdown('<div class="sub-header">📊 Configuration</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
        
        st.markdown('<div class="sub-header">⚙️ Paramètres généraux</div>', unsafe_allow_html=True)
        
        st.session_state['test_size'] = st.slider(
            "Taille ensemble de test (%)", 
            min_value=10, max_value=40, value=20, step=5
        ) / 100
        
        st.session_state['epochs'] = st.slider(
            "Nombre d'époques d'entraînement", 
            min_value=5, max_value=50, value=10, step=5
        )
        
        st.session_state['balance_data'] = st.checkbox(
            "Équilibrer les données", 
            value=st.session_state.get('balance_data', False),
            help="Sous-échantillonne les transactions non frauduleuses pour réduire le déséquilibre des classes"
        )
        
        st.markdown('<div class="sub-header">🧠 Paramètres des modèles</div>', unsafe_allow_html=True)
        
        # Paramètres MLP
        st.markdown("##### Réseau de neurones (MLP)")
        
        mlp_layers = st.text_input(
            "Architecture (neurones par couche)", 
            value="32,16",
            help="Séparés par des virgules, ex: 32,16,8"
        )
        st.session_state['mlp_neurons'] = [int(n) for n in mlp_layers.split(",") if n.strip().isdigit()]
        
        st.session_state['dropout_rate'] = st.slider(
            "Taux de dropout", 
            min_value=0.0, max_value=0.5, value=0.2, step=0.1
        )
        
        # Paramètres Autoencodeur
        st.markdown("##### Autoencodeur")
        
        encoding_layers = st.text_input(
            "Architecture (dimensions encodage)", 
            value="16,8",
            help="Séparés par des virgules, ex: 16,8,4"
        )
        st.session_state['encoding_dim'] = [int(n) for n in encoding_layers.split(",") if n.strip().isdigit()]
        
        st.session_state['custom_threshold'] = st.checkbox(
            "Seuil personnalisé", 
            value=st.session_state.get('custom_threshold', False),
            help="Définir manuellement le seuil d'anomalie"
        )
        
        if st.session_state['custom_threshold']:
            st.session_state['threshold_value'] = st.slider(
                "Seuil d'anomalie (percentile)", 
                min_value=90, max_value=99, value=95, step=1
            ) / 100
        
        # Boutons de lancement
        col1, col2 = st.columns(2)
        with col1:
            start_mlp = st.button("Lancer MLP", use_container_width=True)
        with col2:
            start_autoencoder = st.button("Lancer Autoencodeur", use_container_width=True)
        
        compare_button = st.button("Comparer les modèles", type="primary", use_container_width=True)

    # Corps principal
    data = load_data(uploaded_file)

    if data is not None:
        # Organisation en onglets
        tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🔍 Prédiction", "📈 Comparaison"])
        
        with tab1:
            show_data_info(data)
            
            # Afficher des analyses supplémentaires
            st.markdown('<div class="sub-header">Analyses avancées</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Corrélation avec la classe de fraude
                correlations = data.corr()['Class'].sort_values(ascending=False)
                correlations = correlations.drop('Class')
                
                fig = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title="Corrélation avec la variable cible (Fraude)",
                    color=correlations.values,
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(xaxis_title="Variable", yaxis_title="Coefficient de corrélation")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Montants moyens par type de transaction
                fig = px.box(
                    data,
                    x="Class",
                    y="Amount",
                    color="Class",
                    points="all",
                    labels={"Class": "Type", "Amount": "Montant"},
                    category_orders={"Class": [0, 1]},
                    color_discrete_map={0: "#3366ff", 1: "#ff3366"},
                    title="Distribution des montants par type de transaction"
                )
                fig.update_layout(xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Légitime", "Fraude"]))
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if start_mlp:
                with st.spinner("🔄 Prétraitement et entraînement du MLP en cours..."):
                    X_train, X_test, y_train, y_test, X_test_raw = preprocess_data(data)
                    mlp, mlp_history = train_mlp(X_train, y_train)
                    preds = mlp.predict(X_test).flatten()
                    preds_binary = (preds > 0.5).astype(int)
                    display_results(y_test, preds_binary, preds, "MLP", X_test_raw)
            
            if start_autoencoder:
                with st.spinner("🔄 Prétraitement et entraînement de l'Autoencodeur en cours..."):
                    X_train, X_test, y_train, y_test, X_test_raw = preprocess_data(data)
                    autoencoder, ae_history = train_autoencoder(X_train)
                    preds_binary, reconstruction_error, threshold = get_autoencoder_predictions(autoencoder, X_test, y_test)
                    display_results(y_test, preds_binary, reconstruction_error, "Autoencodeur", X_test_raw, threshold)
        
        with tab3:
            if compare_button or ('mlp_results' in st.session_state and 'autoencoder_results' in st.session_state):
                compare_models()
    else:
        # Guide de démarrage rapide
        st.markdown("""
        ## 👋 Bienvenue dans l'application de détection de fraude bancaire
        
        Cette application vous permet de détecter les transactions frauduleuses à l'aide de deux approches de Deep Learning :
        
        1. **Réseau de neurones (MLP)** - Une approche supervisée classique
        2. **Autoencodeur** - Une approche non supervisée basée sur la détection d'anomalies
        
        ### 🚀 Pour commencer :
        1. Utilisez le panneau latéral pour importer votre fichier CSV de transactions
        2. Explorez les données dans l'onglet "Exploration"
        3. Configurez les paramètres des modèles selon vos besoins
        4. Lancez l'entraînement des modèles et analysez les résultats
        5. Comparez les performances pour choisir la meilleure approche
        
        ### 📊 Format de données attendu :
        - Un fichier CSV avec une colonne 'Class' (0 pour transactions légitimes, 1 pour fraudes)
        - Une colonne 'Amount' pour le montant des transactions
        - D'autres colonnes représentant les caractéristiques des transactions
        
        > 💡 **Conseil** : Vous pouvez utiliser le jeu de données "Credit Card Fraud Detection" de Kaggle comme exemple.
        """)
        
        # Affichage d'un exemple de dashboard avec des données simulées
        st.markdown('<div class="sub-header">Aperçu de la visualisation</div>', unsafe_allow_html=True)
        
        # Simuler des données et un graphique pour montrer un aperçu
        fig = go.Figure()
        x = ['Précision', 'Rappel', 'F1-Score', 'Exactitude']
        y1 = [0.92, 0.83, 0.87, 0.98]
        y2 = [0.88, 0.90, 0.89, 0.97]
        
        fig.add_trace(go.Bar(x=x, y=y1, name='MLP', marker_color='#3366ff'))
        fig.add_trace(go.Bar(x=x, y=y2, name='Autoencodeur', marker_color='#ff3366'))
        
        fig.update_layout(
            title='Exemple de comparaison de modèles',
            xaxis=dict(title='Métrique'),
            yaxis=dict(title='Score'),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()