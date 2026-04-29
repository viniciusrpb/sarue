import json
import random
import re
import unicodedata

TARGET_SIZE = 3000

# =========================
# NORMALIZAÇÃO
# =========================
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# LISTAS
# =========================
KEYWORDS = [
    "epidemiologia", "epidemia", "pandemia", "surto",
    "doencas tropicais", "dengue", "zika", "chikungunya", "malaria",
    "vacinacao", "imunizacao", "vacina", "unidade basica de saude",
    "farmacia popular", "estado de sitio", "estado de emergencia",
    "estado de calamidade", "desastre natural", "enchente", "inundacao",
    "infraestrutura de saude", "deslizamento", "risco geologico",
    "atencao primaria", "atencao basica", "saude da familia", "ubs"
]

LOCALIDADES_DF = [
    "plano piloto","asa sul","asa norte","gama","taguatinga","brazlandia",
    "sobradinho","planaltina","paranoa","nucleo bandeirante","ceilandia",
    "guara","cruzeiro","samambaia","santa maria","sao sebastiao",
    "recanto das emas","lago sul","riacho fundo","lago norte",
    "candangolandia","aguas claras","vicente pires","fercal",
    "sol nascente","por do sol","arniqueira","arapoanga","agua quente",
    "ubs","hospital","upa","unidade basica de saude"
]

# =========================
# DETECTORES
# =========================
def has_location(q):
    q = normalize(q)
    return any(loc in q for loc in LOCALIDADES_DF)

def has_keyword(q):
    q = normalize(q)
    return any(k in q for k in KEYWORDS)

# =========================
# DEDUP
# =========================
def deduplicate(pairs):
    seen = set()
    unique = []

    for p in pairs:
        q = normalize(p.get("pergunta", ""))
        if not q:
            continue
        if q not in seen:
            seen.add(q)
            unique.append(p)

    return unique

# =========================
# CLASSIFICAÇÃO
# =========================
def classify(pairs):
    loc = []
    kw = []
    other = []

    for p in pairs:
        q = p.get("pergunta", "")

        if has_location(q):
            loc.append(p)
        elif has_keyword(q):
            kw.append(p)
        else:
            other.append(p)

    return loc, kw, other

# =========================
# LOAD
# =========================
def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("data", [])
    return data

multi = deduplicate(load("data/dataset_qa_multi.json"))
single = deduplicate(load("data/dataset_qa_single.json"))

# =========================
# CLASSIFICAR
# =========================
m_loc, m_kw, m_other = classify(multi)
s_loc, s_kw, s_other = classify(single)

# embaralhar
for lst in [m_loc, m_kw, m_other, s_loc, s_kw, s_other]:
    random.shuffle(lst)

# =========================
# SELEÇÃO PRIORITÁRIA
# =========================
final = []

def add_from(lst):
    for p in lst:
        if len(final) >= TARGET_SIZE:
            break
        final.append(p)

# prioridade
add_from(m_loc)    # 1. multi + localização
add_from(s_loc)    # 2. single + localização

add_from(m_kw)     # 3. multi + keyword
add_from(s_kw)     # 4. single + keyword

add_from(m_other)  # 5. resto multi
add_from(s_other)  # 6. resto single

# dedup final (segurança)
final = deduplicate(final)[:TARGET_SIZE]

# =========================
# SAVE
# =========================
with open("dataset_final.json", "w", encoding="utf-8") as f:
    json.dump({"data": final}, f, ensure_ascii=False, indent=2)

# =========================
# LOG
# =========================
print("Total:", len(final))
print("Localidades:", sum(has_location(p["pergunta"]) for p in final))
print("Keywords:", sum(has_keyword(p["pergunta"]) for p in final))
