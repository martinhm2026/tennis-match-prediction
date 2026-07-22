from flask import Flask, render_template, request, jsonify
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree
import pandas as pd
import openpyxl
from collections import Counter


app = Flask(__name__)

metrics_data = {
    "Regresión Logística": {
        "Accuracy":  88.46,
        "Precision": 90.00,
        "Recall":    81.82,
        "F1-Score":  85.71,
        "ROC AUC":   93.94
    },
    "Random Forest": {
        "Accuracy":  84.62,
        "Precision": 81.82,
        "Recall":    81.82,
        "F1-Score":  81.82,
        "ROC AUC":   88.79
    },
    "SVM (RBF)": {
        "Accuracy":  80.77,
        "Precision": 87.50,
        "Recall":    63.64,
        "F1-Score":  73.68,
        "ROC AUC":   87.27
    }
}


# ---------------------------------------------------------
# 1) Carpeta de modelos y datos
# ---------------------------------------------------------
model_folder = os.path.join(app.root_path, 'Modelo')

# ---------------------------------------------------------
# 2) Carga de pickles y scaler
# ---------------------------------------------------------
with open(os.path.join(model_folder, "modelo_log.pkl"), "rb") as f:
    modelo_log = pickle.load(f)
with open(os.path.join(model_folder, "modelo_svm.pkl"), "rb") as f:
    modelo_svm = pickle.load(f)
with open(os.path.join(model_folder, "modelo_rf.pkl"), "rb") as f:
    modelo_rf = pickle.load(f)
with open(os.path.join(model_folder, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------------
# 3) Carga de estadísticas de jugadoras
# ---------------------------------------------------------
df_jugadoras = pd.read_excel(
    os.path.join(model_folder, "estadisticas_jugadoras.xlsx")
)
df_partidos = pd.read_csv(
    os.path.join(model_folder, "2. Partidos de Tenis.csv")
)

# ---------------------------------------------------------
# 4) Lógica de predicción
# ---------------------------------------------------------
def predecir_resultado(jugadora_A, jugadora_B):
    # Extraemos las estadísticas de cada jugadora
    try:
        stats_A = (
            df_jugadoras[df_jugadoras["Jugadora"] == jugadora_A]
            .drop(columns="Jugadora")
            .squeeze()
        )
        stats_B = (
            df_jugadoras[df_jugadoras["Jugadora"] == jugadora_B]
            .drop(columns="Jugadora")
            .squeeze()
        )
    except Exception:
        return {"error": "Una o ambas jugadoras no existen en el dataset."}

    # Construimos dos instancias: A vs B y B vs A
    inst1 = (stats_A - stats_B).values.reshape(1, -1)
    inst2 = (stats_B - stats_A).values.reshape(1, -1)

    # Escalamos
    X1 = scaler.transform(inst1)
    X2 = scaler.transform(inst2)

    modelos = {
        "Regresión Logística": modelo_log,
        "SVM (RBF)":           modelo_svm,
        "Random Forest":       modelo_rf
    }

    resultados = []
    for nombre, modelo in modelos.items():
        p1 = modelo.predict_proba(X1)[0]
        p2 = modelo.predict_proba(X2)[0]

        proba_A = (p1[1] + p2[0]) / 2
        proba_B = (p1[0] + p2[1]) / 2
        ganadora = jugadora_A if proba_A > proba_B else jugadora_B

        # Buscamos métricas precomputadas
        m = metrics_data[nombre]
        resultados.append({
            "Modelo":            nombre,
            "Ganadora_Predicha": ganadora,
            "Accuracy":          m["Accuracy"],
            "Precision":         m["Precision"],
            "Recall":            m["Recall"],
            "F1-Score":          m["F1-Score"],
            "ROC AUC":           m["ROC AUC"]
        })

    return resultados
players = [
    {"name": "Su-Wei Hsieh",               "value": "su-wei_hsieh",               "country": "tpe"},
    {"name": "Eugenie Bouchard",           "value": "eugenie_bouchard",           "country": "can"},
    {"name": "Jie Zheng",                  "value": "jie_zheng",                  "country": "chn"},
    {"name": "Tamira Paszek",              "value": "tamira_paszek",              "country": "aut"},
    {"name": "Karin Knapp",                "value": "karin_knapp",                "country": "ita"},
    {"name": "Vania King",                 "value": "vania_king",                 "country": "usa"},
    {"name": "Marina Erakovic",            "value": "marina_erakovic",            "country": "nzl"},
    {"name": "Dominika Cibulkova",         "value": "dominika_cibulkova",         "country": "svk"},
    {"name": "Kimiko Date-Krumm",          "value": "kimiko_date-krumm",          "country": "jpn"},
    {"name": "Kristina Mladenovic",        "value": "kristina_mladenovic",        "country": "fra"},
    {"name": "Karolina Pliskova",          "value": "karolina_pliskova",          "country": "cze"},
    {"name": "Jelena Jankovic",            "value": "jelena_jankovic",            "country": "srb"},
    {"name": "Jamie Hampton",              "value": "jamie_hampton",              "country": "usa"},
    {"name": "Anna Karolina Schmiedlova",  "value": "anna_karolina_schmiedlova",  "country": "svk"},
    {"name": "Shuai Peng",                 "value": "shuai_peng",                 "country": "chn"},
    {"name": "Petra Kvitova",              "value": "petra_kvitova",              "country": "cze"},
    {"name": "Elena Vesnina",              "value": "elena_vesnina",              "country": "rus"},
    {"name": "Sandra Zahlavova",           "value": "sandra_zahlavova",           "country": "cze"},
    {"name": "Irina-Camelia Begu",         "value": "irina-camelia_begu",         "country": "rou"},
    {"name": "Alize Cornet",               "value": "alize_cornet",               "country": "fra"},
    {"name": "Flavia Pennetta",            "value": "flavia_pennetta",            "country": "ita"},
    {"name": "Melinda Czink",              "value": "melinda_czink",              "country": "hun"},
    {"name": "Kristyna Pliskova",          "value": "kristyna_pliskova",          "country": "cze"},
    {"name": "Marion Bartoli",             "value": "marion_bartoli",             "country": "fra"},
    {"name": "Nina Bratchikova",           "value": "nina_bratchikova",           "country": "rus"},
    {"name": "Ashleigh Barty",             "value": "ashleigh_barty",             "country": "aus"},
    {"name": "Stefanie Voegele",           "value": "stefanie_voegele",           "country": "sui"},
    {"name": "Klara Zakopalova",           "value": "klara_zakopalova",           "country": "cze"},
    {"name": "Coco Vandeweghe",            "value": "coco_vandeweghe",            "country": "usa"},
    {"name": "Tatjana Maria",              "value": "tatjana_maria",              "country": "ger"},
    {"name": "Bethanie Mattek-Sands",      "value": "bethanie_mattek-sands",      "country": "usa"},
    {"name": "Na Li",                      "value": "na_li",                      "country": "chn"},
    {"name": "Arantxa Rus",                "value": "arantxa_rus",                "country": "ned"},
    {"name": "Ayumi Morita",               "value": "ayumi_morita",               "country": "jpn"},
    {"name": "Maria-Teresa Torro-Flor",    "value": "maria-teresa_torro-flor",    "country": "esp"},
    {"name": "Sabine Lisicki",             "value": "sabine_lisicki",             "country": "ger"},
    {"name": "Simona Halep",               "value": "simona_halep",               "country": "rou"},
    {"name": "Irena Pavlovic",             "value": "irena_pavlovic",             "country": "fra"},
    {"name": "Madison Keys",               "value": "madison_keys",               "country": "usa"},
    {"name": "Nadia Petrova",              "value": "nadia_petrova",              "country": "rus"},
    {"name": "Petra Martic",               "value": "petra_martic",               "country": "cro"},
    {"name": "Chanelle Scheepers",         "value": "chanelle_scheepers",         "country": "rsa"},
    {"name": "Virginie Razzano",           "value": "virginie_razzano",           "country": "fra"},
    {"name": "Julia Goerges",              "value": "julia_goerges",              "country": "ger"},
    {"name": "Urszula Radwanska",          "value": "urszula_radwanska",          "country": "pol"},
    {"name": "Dinah Pfizenmaier",          "value": "dinah_pfizenmaier",          "country": "ger"},
    {"name": "Mallory Burdette",           "value": "mallory_burdette",           "country": "usa"},
    {"name": "Agnieszka Radwanska",        "value": "agnieszka_radwanska",        "country": "pol"},
    {"name": "Mona Barthel",               "value": "mona_barthel",               "country": "ger"},
    {"name": "Jana Cepelova",              "value": "jana_cepelova",              "country": "svk"},
    {"name": "Romina Oprandi",             "value": "romina_oprandi",             "country": "sui"},
    {"name": "Varvara Lepchenko",          "value": "varvara_lepchenko",          "country": "usa"},
    {"name": "Svetlana Kuznetsova",        "value": "svetlana_kuznetsova",        "country": "rus"},
    {"name": "Pauline Parmentier",         "value": "pauline_parmentier",         "country": "fra"},
    {"name": "Barbora Zahlavova Strycova", "value": "barbora_zahlavova_strycova", "country": "cze"},
    {"name": "Caroline Wozniacki",         "value": "caroline_wozniacki",         "country": "den"},
    {"name": "Stephanie Foretz Gacon",     "value": "stephanie_foretz_gacon",     "country": "fra"},
    {"name": "Galina Voskoboeva",          "value": "galina_voskoboeva",          "country": "kaz"},
    {"name": "Petra Cetkovska",            "value": "petra_cetkovska",            "country": "cze"},
    {"name": "Anastasia Pavlyuchenkova",   "value": "anastasia_pavlyuchenkova",   "country": "rus"},
    {"name": "Kiki Bertens",               "value": "kiki_bertens",               "country": "ned"},
    {"name": "Monica Niculescu",           "value": "monica_niculescu",           "country": "rou"},
    {"name": "Yuliya Beygelzimer",         "value": "yuliya_beygelzimer",         "country": "ukr"},
    {"name": "Serena Williams",            "value": "serena_williams",            "country": "usa"},
    {"name": "Melanie Oudin",              "value": "melanie_oudin",              "country": "usa"},
    {"name": "Annika Beck",                "value": "annika_beck",                "country": "ger"},
    {"name": "Francesca Schiavone",        "value": "francesca_schiavone",        "country": "ita"},
    {"name": "Kaia Kanepi",                "value": "kaia_kanepi",                "country": "est"},
    {"name": "Paula Ormaechea",            "value": "paula_ormaechea",            "country": "arg"},
    {"name": "Yulia Putintseva",           "value": "yulia_putintseva",           "country": "kaz"},
    {"name": "Shelby Rogers",              "value": "shelby_rogers",              "country": "usa"},
    {"name": "Monica Puig",                "value": "monica_puig",                "country": "pur"},
    {"name": "Mathilde Johansson",         "value": "mathilde_johansson",         "country": "fra"},
    {"name": "Zuzana Kucova",              "value": "zuzana_kucova",              "country": "svk"},
    {"name": "Magdalena Rybarikova",       "value": "magdalena_rybarikova",       "country": "svk"},
    {"name": "Johanna Larsson",            "value": "johanna_larsson",            "country": "swe"},
    {"name": "Bojana Jovanovski",          "value": "bojana_jovanovski",          "country": "srb"},
    {"name": "Sloane Stephens",            "value": "sloane_stephens",            "country": "usa"},
    {"name": "Carla Suarez Navarro",       "value": "carla_suarez_navarro",       "country": "esp"},
    {"name": "Maria Kirilenko",            "value": "maria_kirilenko",            "country": "rus"},
    {"name": "Victoria Azarenka",          "value": "victoria_azarenka",          "country": "blr"},
    {"name": "Maria Sharapova",            "value": "maria_sharapova",            "country": "rus"},
    {"name": "Tsvetana Pironkova",         "value": "tsvetana_pironkova",         "country": "bul"},
    {"name": "Vesna Dolonc",               "value": "vesna_dolonc",               "country": "srb"},
    {"name": "Alexandra Cadantu",          "value": "alexandra_cadantu",          "country": "rou"},
    {"name": "Elena Baltacha",             "value": "elena_baltacha",             "country": "gbr"},
    {"name": "Lesia Tsurenko",             "value": "lesia_tsurenko",             "country": "ukr"},
    {"name": "Samantha Stosur",            "value": "samantha_stosur",            "country": "aus"},
    {"name": "Lauren Davis",               "value": "lauren_davis",               "country": "usa"},
    {"name": "Garbine Muguruza",           "value": "garbine_muguruza",           "country": "esp"},
    {"name": "Daniela Hantuchova",         "value": "daniela_hantuchova",         "country": "svk"},
    {"name": "Lucie Safarova",             "value": "lucie_safarova",             "country": "cze"},
    {"name": "Yanina Wickmayer",           "value": "yanina_wickmayer",           "country": "bel"},
    {"name": "Camila Giorgi",              "value": "camila_giorgi",              "country": "ita"},
    {"name": "Aravane Rezai",              "value": "aravane_rezai",              "country": "fra"},
    {"name": "Silvia Soler-Espinosa",      "value": "silvia_soler-espinosa",      "country": "esp"},
    {"name": "Maria Joao Koehler",         "value": "maria_joao_koehler",         "country": "por"},
    {"name": "Kirsten Flipkens",           "value": "kirsten_flipkens",           "country": "bel"},
    {"name": "Mariana Duque-Marino",       "value": "mariana_duque-marino",       "country": "col"},
    {"name": "Olga Govortsova",            "value": "olga_govortsova",            "country": "blr"},
    {"name": "Lucie Hradecka",             "value": "lucie_hradecka",             "country": "cze"},
    {"name": "Heather Watson",             "value": "heather_watson",             "country": "gbr"},
    {"name": "Yaroslava Shvedova",         "value": "yaroslava_shvedova",         "country": "kaz"},
    {"name": "Lourdes Dominguez Lino",     "value": "lourdes_dominguez_lino",     "country": "esp"},
    {"name": "A. Medina Garrigues",        "value": "a._medina_garrigues",        "country": "esp"},
    {"name": "Sara Errani",                "value": "sara_errani",                "country": "ita"},
    {"name": "Julia Glushko",              "value": "julia_glushko",              "country": "isr"},
    {"name": "Sofia Arvidsson",            "value": "sofia_arvidsson",            "country": "swe"},
    {"name": "Misaki Doi",                 "value": "misaki_doi",                 "country": "jpn"},
    {"name": "Ana Ivanovic",               "value": "ana_ivanovic",               "country": "srb"},
    {"name": "Claire Feuerstein",          "value": "claire_feuerstein",          "country": "fra"},
    {"name": "Venus Williams",             "value": "venus_williams",             "country": "usa"},
    {"name": "Mandy Minella",              "value": "mandy_minella",              "country": "lux"},
    {"name": "Donna Vekic",                "value": "donna_vekic",                "country": "cro"},
    {"name": "Shahar Peer",                "value": "shahar_peer",                "country": "isr"},
    {"name": "Angelique Kerber",           "value": "angelique_kerber",           "country": "ger"},
    {"name": "Christina McHale",           "value": "christina_mchale",           "country": "usa"},
    {"name": "Elina Svitolina",            "value": "elina_svitolina",            "country": "ukr"},
    {"name": "Mirjana Lucic-Baroni",       "value": "mirjana_lucic-baroni",       "country": "cro"},
    {"name": "Ekaterina Makarova",         "value": "ekaterina_makarova",         "country": "rus"},
    {"name": "Laura Robson",               "value": "laura_robson",               "country": "gbr"},
    {"name": "Roberta Vinci",              "value": "roberta_vinci",              "country": "ita"},
    {"name": "Grace Min",                  "value": "grace_min",                  "country": "usa"},
    {"name": "Olga Puchkova",              "value": "olga_puchkova",              "country": "rus"},
    {"name": "Andrea Hlavackova",          "value": "andrea_hlavackova",          "country": "cze"},
    {"name": "Sorana Cirstea",             "value": "sorana_cirstea",             "country": "rou"},
    {"name": "Caroline Garcia",            "value": "caroline_garcia",            "country": "fra"},
    {"name": "Anna Tatishvili",            "value": "anna_tatishvili",            "country": "usa"}
]

# ---------------------------------------------------------
# 5) Endpoint JSON para simulación
# ---------------------------------------------------------
@app.route("/simular", methods=["POST"])
def simular():
    data = request.get_json(force=True)
    j1 = data.get("jugadora1")
    j2 = data.get("jugadora2")

    if not j1 or not j2:
        return jsonify({"error": "Faltan nombres de jugadoras"}), 400

    resultados = predecir_resultado(j1, j2)
    # Si la función devolvió un dict de error:
    if isinstance(resultados, dict) and resultados.get("error"):
        return jsonify(resultados), 400

    return jsonify(resultados), 200


@app.route('/')
def inicio():
    # Lista de creadores: cada dict tiene nombre e imagen
    creators = [
        {"name": "Alvaro Arce",   "avatar": "icono.png", "flag": "chile.png"},
        {"name": "Gabriela Blanco",    "avatar": "icono.png", "flag": "chile.png"},
        {"name": "Martin Huerta",   "avatar": "icono.png", "flag": "chile.png"},
        {"name": "Sebastián Tello", "avatar": "icono.png", "flag": "chile.png"},
        {"name": "Cristobal Vargas","avatar": "icono.png", "flag": "chile.png"},
        {"name": "Paz Vivar","avatar": "icono.png", "flag": "chile.png"}
    ]
    return render_template('inicio.html', creators=creators)

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():    
    return render_template(
        'prediccion.html',
        players=players
    )

@app.route('/ranking')
def ranking():
    return render_template('ranking.html')

@app.route('/stats_jugadora', methods=['POST'])
def stats_jugadora():
    data = request.get_json(force=True)
    jug = data.get('jugadora')
    row = df_jugadoras[df_jugadoras['Jugadora'] == jug]
    if row.empty:
        return jsonify({'error': 'Jugadora no encontrada'}), 404
    stats = row.iloc[0].to_dict()
    stats['Jugadora'] = jug
    return jsonify(stats)

@app.route('/matches_jugadora', methods=['POST'])
def matches_jugadora():
    data = request.get_json(force=True)
    jug = data.get('jugadora')
    mask = (df_partidos['Player1'] == jug) | (df_partidos['Player2'] == jug)
    recientes = df_partidos[mask].tail(3)
    resultados = []
    for _, row in recientes.iterrows():
        # Contar sets ganados
        sets1 = sum(
            1 for i in range(1, 6)
            if row[f'ST{i}.1'] > row[f'ST{i}.2']
        )
        sets2 = sum(
            1 for i in range(1, 6)
            if row[f'ST{i}.2'] > row[f'ST{i}.1']
        )
        p1 = next(p for p in players if p['name'] == row['Player1'])
        p2 = next(p for p in players if p['name'] == row['Player2'])
        resultados.append({
            'Player1': {'name': row['Player1'], 'value': p1['value'], 'country': p1['country']},
            'Player2': {'name': row['Player2'], 'value': p2['value'], 'country': p2['country']},
            'sets1': sets1,
            'sets2': sets2
        })
    return jsonify(resultados)

@app.route('/Individual')
def Individual():
    return render_template('Individual.html', players=players)

if __name__ == '__main__':
    app.run(debug=True)
