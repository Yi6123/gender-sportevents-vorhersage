import streamlit as st
import pandas as pd
import joblib

# Seitentitel
st.title("🎯 Vorhersage der Geschlechterdominanz in Sportarten")
st.markdown("Geben Sie ein Jahr und eine Sportart ein, um vorherzusagen, ob diese männlich, weiblich oder ausgewogen dominiert ist.")

# Modelle und Mappings laden
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
sport_mapping = joblib.load("sport_mapping.pkl")
label_mapping = joblib.load("gender_dominance_mapping.pkl")

# Benutzereingabe
year = st.number_input("Bitte geben Sie ein Jahr ein (1976–2024):", min_value=1976, max_value=2024, value=2000, step=4)

available_sports = sorted(list(sport_mapping.keys()))
sport = st.selectbox("Bitte wählen Sie eine Sportart:", available_sports)

# Vorhersage
if st.button("Vorhersage starten"):

    # 1. Jahr skalieren
    year_scaled = scaler.transform([[year]])[0][0]

    # 2. One-hot Input vorbereiten
    feature_columns = list(model.feature_names_in_)
    input_data = dict.fromkeys(feature_columns, 0)
    input_data["Year"] = year_scaled
    input_data[sport_mapping[sport]] = 1

    # 3. In DataFrame umwandeln
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    # 4. Vorhersage durchführen
    prediction = model.predict(input_df)[0]
    readable_result = label_mapping[prediction]

    # 5. Ergebnis anzeigen
    st.success(f"Ergebnis:{readable_result}")