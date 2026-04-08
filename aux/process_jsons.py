import json
import os
import re
from collections import defaultdict

ubses = [
    "UBS 1 de Águas Claras", "UBS 2 de Águas Claras", "UBS 1 da Asa Norte", "UBS 2 da Asa Norte", "UBS 1 da Asa Sul", "UBS 1 de Brazlândia", "UBS 2 de Brazlândia",
    "UBS 1 do Gama", "UBS 2 do Gama", "UBS 1 do Cruzeiro"
]

def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower())

dic = defaultdict(list)

path = "UBS/"

for filename in os.listdir(path):
    print(filename)
    if filename.endswith(".json"):
        filepath = os.path.join(path, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                dodf_data = json.load(f)

            for item_id, materia in dodf_data.items():
                texto = materia.get("texto", "")
                titulo = materia.get("titulo", "")
                texto2 = normalize(texto)

                print(texto2)

                for ubs in ubses:
                    if normalize(ubs) in texto2:
                        dic[ubs].append({
                            "arquivo": filename,
                            "id": item_id.strip(),
                            "titulo": titulo,
                            "trecho": texto[:500]
                        })
                        break

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

output = "database.json"
output_file = open(output, "w", encoding="utf-8")
json.dump(dic, output_file, ensure_ascii=False, indent=2)
