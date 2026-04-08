import pandas as pd

def load_documents():

    paths = ["noticias_ses_1_100.csv", "fiocruz_noticias.csv", "min_saude.csv"]

    docs = []

    for path in paths:
        df = pd.read_csv("samples/"+path)

        for _, row in df.iterrows():
            titulo = str(row.get("titulo", "")).strip()
            noticia = str(row.get("noticia", "")).strip()

            text = f"{titulo}\n\n{noticia}".strip()

            if len(text) > 50:
                docs.append(text)

    return docs
