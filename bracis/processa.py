"""
processar_dodf.py
-----------------
Lê uma pasta com JSONs do Diário Oficial do Distrito Federal (DODF),
filtra publicações da "Secretaria de Estado de Saúde" (em todas as seções),
e consolida tudo em um único arquivo JSON organizado por dia.

Estrutura de saída:
{
  "2025-05-08": {
    "edicao": 84,
    "publicacoes": [
      {
        "secao": "Seção I",
        "tipo_documento": "Portaria conjunta",
        "numero_documento": "PORTARIA CONJUNTA Nº 18, DE 09 DE ABRIL DE 2025",
        "texto_completo": "..."
      },
      ...
    ]
  },
  ...
}

Uso:
    python processar_dodf.py --pasta ./jsons --saida resultado.json

    # Filtrar por nome diferente de órgão:
    python processar_dodf.py --pasta ./jsons --saida resultado.json \
        --orgao "Secretaria de Estado de Saúde"
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
import unicodedata

KEYWORDS = [
    "epidemiologia", "epidemia", "pandemia", "surto",
    "doenças tropicais", "dengue", "zika", "chikungunya", "malária",
    "vacinação", "imunização", "vacina","unidade básica de saúde","farmácia popular",
    "estado de sítio", "estado de emergencia", "estado de calamidade",
    "desastre natural", "enchente", "inundação","infraestrutura de saúde",
    "deslizamento", "risco geológico",
    "atenção primária", "atenção básica", "saúde da família", "ubs"
]

LOCALIDADES_DF = [
    "plano piloto",
    "asa sul",
    "asa norte",
    "vila planalto",
    "noroeste",
    "gama",
    "taguatinga",
    "brazlandia",
    "sobradinho",
    "planaltina",
    "paranoa",
    "nucleo bandeirante",
    "ceilandia",
    "guara",
    "cruzeiro",
    "samambaia",
    "santa maria",
    "cafe sem troco",
    "padf",
    "sao sebastiao",
    "recanto das emas",
    "lago sul",
    "riacho fundo",
    "lago norte",
    "candangolandia",
    "aguas claras",
    "riacho fundo ii",
    "sudoeste",
    "octogonal",
    "varjao",
    "park way",
    "scia",
    "setor complementar de industria e abastecimento",
    "sobradinho ii",
    "jardim botanico",
    "itapoa",
    "sia",
    "setor de industria e abastecimento",
    "vicente pires",
    "fercal",
    "sol nascente",
    "por do sol",
    "arniqueira",
    "arapoanga",
    "agua quente"
]

TIPOS_UNIDADE = [
    "ubs", "unidade basica de saude", "unidade basica", "hospital", "unidade de pronto atendimento", "upa"
]

PROGRAMAS = [
    "farmacia popular", "saude da familia", "estratégia saude da familia", "saude bucal"
]


def extrair_localidades(texto):
    t = normalizar(texto)
    encontrados = []

    for loc in LOCALIDADES_DF:
        pattern = r"\b" + re.escape(loc) + r"\b"
        if re.search(pattern, t):
            encontrados.append(loc)

    return list(set(encontrados))


def extrair_unidades(texto):
    t = normalizar(texto)
    return [u for u in TIPOS_UNIDADE if u in t]


def extrair_programas(texto):
    t = normalizar(texto)
    return [p for p in PROGRAMAS if p in t]


def extrair_topicos(texto):
    t = normalizar(texto)
    encontrados = []

    for k in KEYWORDS_NORM:
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            encontrados.append(k)

    return encontrados

class _HTMLStripper(HTMLParser):

    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(p.strip() for p in self._parts if p.strip())

def strip_html(html):
    if not html:
        return ""
    parser = _HTMLStripper()
    parser.feed(html)
    return parser.get_text()


def extrair_data(caminho_arquivo, dados_json):

    match = re.search(r"(\d{4}-\d{2}-\d{2})", caminho_arquivo.name)
    if match:
        return match.group(1)

    j = dados_json.get("diario", {}).get("json", {})
    dt = j.get("dt_publicacao")
    if dt:

        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(dt, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass

    titulo = j.get("ds_titulo", "")
    meses = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
        "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
        "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    }
    m = re.search(
        r"(\d{1,2})\s+DE\s+(\w+)\s+DE\s+(\d{4})",
        titulo,
        re.IGNORECASE,
    )
    if m:
        dia, mes_nome, ano = m.group(1), m.group(2).lower(), m.group(3)
        if mes_nome in meses:
            try:
                return datetime(int(ano), meses[mes_nome], int(dia)).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None




def normalizar(texto):
    if not texto:
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


KEYWORDS_NORM = [normalizar(k) for k in KEYWORDS]


def contem_palavra_chave(texto):
    texto_norm = normalizar(texto)

    for k in KEYWORDS_NORM:
        pattern = r"\b" + re.escape(k) + r"\b"
        if re.search(pattern, texto_norm):
            print("MATCH KEYWORD:", k)
            return True

    return False

ORGAO_PADRAO = "Secretaria de Estado de Saúde"
CO_DEMANDANTE_ALVO = "782"
def extrair_publicacoes_arquivo(caminho, orgao_alvo=ORGAO_PADRAO):

    with caminho.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    data_str = extrair_data(caminho, raw)

    # --------------------------------------------------
    # 🔎 Detecta formato do JSON
    # --------------------------------------------------
    if "json" in raw and isinstance(raw["json"], dict):
        diario_json = raw["json"]  # formato novo (API)
    elif "diario" in raw:
        diario_json = raw.get("diario", {}).get("json") or raw.get("diario")
    else:
        diario_json = raw

    # --------------------------------------------------
    # 🧾 Edição
    # --------------------------------------------------
    edicao = diario_json.get("nu_numero") or diario_json.get("nu_jornal")

    try:
        edicao = int(edicao) if edicao is not None else None
    except (TypeError, ValueError):
        edicao = None

    publicacoes = []

    # ==================================================
    # 🟢 CASO 1 — FORMATO ANTIGO (com INFO)
    # ==================================================
    info = diario_json.get("INFO")

    if isinstance(info, dict):

        for nome_secao, orgaos in info.items():
            if not isinstance(orgaos, dict):
                continue

            secao_norm = normalizar(nome_secao)

            for nome_orgao, conteudo_orgao in orgaos.items():

                orgao_norm = normalizar(nome_orgao)
                alvo_norm = normalizar(orgao_alvo)

                documentos = conteudo_orgao.get("documentos", {})
                if not isinstance(documentos, dict):
                    continue

                for _id, doc in documentos.items():
                    if not isinstance(doc, dict):
                        continue

                    tipo = doc.get("tipo", "").strip()
                    titulo = doc.get("titulo", "").strip()
                    texto_html = doc.get("texto")

                    if not texto_html:
                        print(f"[AVISO] publicação encontrada, mas texto ausente ({caminho.name})")
                        texto = ""
                    else:
                        texto = strip_html(texto_html)

                    texto_base = f"{titulo} {texto}"

                    incluir = False

                    # ✔ REGRA 1 — Seção I com palavras-chave
                    if secao_norm == "secao i":
                        if contem_palavra_chave(texto_base):
                            incluir = True

                    # ✔ REGRA 2 — Seção III com SES
                    elif secao_norm == "secao iii":
                        if alvo_norm in orgao_norm:
                            incluir = True

                    if not incluir:
                        continue

                    publicacoes.append(
                        {
                            "secao": nome_secao,
                            "tipo_documento": tipo,
                            "numero_documento": titulo,
                            "texto_completo": texto,
                            "data": data_str,
                            "topicos": extrair_topicos(texto_base),
                            "localidades": extrair_localidades(texto_base),
                            "tipo_unidade": extrair_unidades(texto_base),
                            "programas": extrair_programas(texto_base),
                        }
                    )

        return data_str, edicao, publicacoes

    # ==================================================
    # 🔴 CASO 2 — FORMATO NOVO (sem INFO)
    # ==================================================
    print(f"[INFO] Formato alternativo detectado: {caminho.name}")

    # tenta localizar estrutura alternativa
    demandantes = diario_json.get("lstHierarquia", {}).get("lstDemandantes", [])

    encontrou_ses = False

    for d in demandantes:
        nome = d.get("ds_nome", "")
        co = str(d.get("co_demandante", ""))

        if "saude" in normalizar(nome) or co == "782":
            encontrou_ses = True

            # ⚠️ neste formato normalmente NÃO há texto aqui
            print(f"[AVISO] publicação encontrada, mas texto ausente ({caminho.name})")

            publicacoes.append(
                {
                    "secao": "Desconhecida",
                    "tipo_documento": "",
                    "numero_documento": f"Registro estrutural: {nome}",
                    "texto_completo": "",
                    "data": data_str,
                    "topicos": [],
                    "localidades": [],
                    "tipo_unidade": [],
                    "programas": [],
                }
            )

    if not encontrou_ses:
        print(f"[INFO] Nenhuma publicação da SES encontrada em {caminho.name}")

    return data_str, edicao, publicacoes

def processar_pasta(pasta, orgao_alvo = ORGAO_PADRAO,verbose = True):

    arquivos = sorted(pasta.glob("*.json"))
    if not arquivos:
        print(f"[AVISO] Nenhum arquivo .json encontrado em: {pasta}", file=sys.stderr)
        return {}

    resultado = {}

    for arq in arquivos:
        if verbose:
            print(f"Processando: {arq.name} …", end=" ", flush=True)

        try:
            data_str, edicao, publicacoes = extrair_publicacoes_arquivo(arq, orgao_alvo)
        except Exception as exc:
            print(f"\n[ERRO] {arq.name}: {exc}", file=sys.stderr)
            continue

        if data_str is None:
            print(
                f"\n[AVISO] Não foi possível determinar a data de {arq.name}; "
                "arquivo ignorado.",
                file=sys.stderr,
            )
            continue

        if data_str not in resultado:
            resultado[data_str] = {"edicao": edicao, "publicacoes": []}

        resultado[data_str]["publicacoes"].extend(publicacoes)

        if verbose:
            print(f"{len(publicacoes)} publicação(ões) encontrada(s).")

    resultado = dict(sorted(resultado.items()))
    return resultado

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filtra publicações do DODF por órgão e consolida em JSON por dia."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pasta",
        type=Path,
        default=Path("."),
        help="Pasta contendo os arquivos .json do DODF (padrão: diretório atual).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resultado_saude.json"),
        help="Arquivo JSON de saída (padrão: resultado_saude.json).",
    )
    parser.add_argument(
        "--orgao",
        type=str,
        default=ORGAO_PADRAO,
        help=f'Nome (parcial) do órgão a filtrar (padrão: "{ORGAO_PADRAO}").',
    )
    parser.add_argument(
        "--silencioso",
        action="store_true",
        help="Suprime mensagens de progresso.",
    )
    args = parser.parse_args()

    if not args.pasta.is_dir():
        print(f"[ERRO] Pasta não encontrada: {args.pasta}", file=sys.stderr)
        sys.exit(1)

    print(f"Órgão alvo : {args.orgao}")
    print(f"Pasta fonte: {args.pasta.resolve()}")
    print(f"Arquivo de saída: {args.saida.resolve()}")
    print("-" * 60)

    resultado = processar_pasta(
        pasta=args.pasta,
        orgao_alvo=args.orgao,
        verbose=not args.silencioso,
    )

    total_pubs = sum(len(v["publicacoes"]) for v in resultado.values())
    print("-" * 60)
    print(f"Total de dias processados : {len(resultado)}")
    print(f"Total de publicações      : {total_pubs}")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\nArquivo salvo em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
