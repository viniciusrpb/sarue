"""
gerar_qa_dodf.py
----------------
Lê o JSON consolidado do DODF (saída de processar_dodf.py) e gera
exatamente 3 000 pares pergunta-resposta usando a API do ChatGPT (OpenAI).

Distribuição:
  • Proporcional ao número de publicações em cada seção (Seção I / II / III)
    dentro de cada dia.
  • Dentro de cada seção, cada publicação recebe pelo menos 1 par;
    os pares extras são distribuídos proporcionalmente ao tamanho do texto.

Estilos de pergunta variados (instruídos no prompt):
  valores financeiros, contratos/licitações, vigências/prazos,
  cargos/servidores, dados normativos gerais, perguntas cruzando
  múltiplas publicações do mesmo dia.

Saída: dataset_qa_dodf.json
  [
    {
      "pergunta": "...",
      "resposta": "...",
      "secao": "Seção II",
      "fonte": "Edição 84 — 2025-05-08"
    },
    ...
  ]

Uso:
    export OPENAI_API_KEY="sk-..."
    python gerar_qa_dodf.py --entrada resultado_saude.json --saida dataset_qa_dodf.json

    # Opções avançadas
    python gerar_qa_dodf.py \\
        --entrada resultado_saude.json \\
        --saida dataset_qa_dodf.json \\
        --total 3000 \\
        --modelo gpt-4o-mini \\
        --pares-por-chamada 5 \\
        --max-workers 5 \\
        --checkpoint checkpoint_qa.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

CUSTO = {
    "gen_input": 0,
    "gen_output": 0,
    "val_input": 0,
    "val_output": 0,
}

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("[ERRO] Instale o SDK da OpenAI:  pip install openai", file=sys.stderr)
    sys.exit(1)

TOTAL_PARES = 2_000
MODELO_PADRAO = "gpt-4o-mini"
PARES_POR_CHAMADA = 5        # quantos pares pedir de uma vez ao modelo
MAX_WORKERS = 5              # chamadas paralelas à API
MAX_RETRIES = 5
BACKOFF_BASE = 2.0           # segundos de espera inicial em retry

ESTILOS = [
    "valor financeiro ou orçamentário (R$, dotação, fonte de recurso)",
    "contrato, licitação ou processo administrativo (número, objeto, partes)",
    "vigência, prazo ou data de publicação/validade",
    "localização geográfica (Região Administrativa, bairro ou endereço mencionado)",
    "identificação de unidade de saúde (UBS, UPA, hospital) e sua localização",
    "objetivo ou ementa geral do ato",
    "pergunta que relacione informações de duas ou mais publicações do mesmo dia/seção",
]

SYSTEM_PROMPT = """\
Você é um especialista em Direito Administrativo e em publicações do \
Diário Oficial do Distrito Federal (DODF). \
Sua tarefa é criar pares de PERGUNTA e RESPOSTA baseados exclusivamente \
no(s) texto(s) fornecido(s). As respostas devem ser precisas e fundamentadas \
no conteúdo das publicações — nunca invente informações ausentes no texto.
"""

USER_TEMPLATE = """\
Abaixo estão {n_pubs} publicação(ões) da {secao} do DODF, \
edição {edicao}, data {data}:

{blocos}

---

Crie EXATAMENTE {n_pares} par(es) de pergunta e resposta com base nesses textos.
Varie os estilos de pergunta conforme a lista a seguir (use cada estilo \
pelo menos uma vez quando houver pares suficientes):
{estilos}

REGRAS:
1. Cada pergunta deve ser objetiva e ter resposta encontrável no(s) texto(s).
2. Prefira perguntas específicas (números, datas, nomes, valores).
3. NÃO repita perguntas entre si.
4. Responda APENAS com um array JSON válido, sem markdown, sem texto extra:
[
  {{"pergunta": "...", "resposta": "..."}},
  ...
]
5. Inclua obrigatoriamente perguntas que envolvam localização geográfica
   (Regiões Administrativas do DF, UBSs, hospitais, etc.) quando essa
   informação estiver presente no texto.
"""

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def imprimir_custo():

    # preços aproximados do gpt-4o-mini
    PRECO_INPUT = 0.00015 / 1000
    PRECO_OUTPUT = 0.0006 / 1000

    custo_gen = (
        CUSTO["gen_input"] * PRECO_INPUT +
        CUSTO["gen_output"] * PRECO_OUTPUT
    )

    custo_val = (
        CUSTO["val_input"] * PRECO_INPUT +
        CUSTO["val_output"] * PRECO_OUTPUT
    )

    custo_total = custo_gen + custo_val

    print("\n====== CUSTO ======")
    print(f"Geração:")
    print(f"  input tokens:  {CUSTO['gen_input']}")
    print(f"  output tokens: {CUSTO['gen_output']}")
    print(f"  custo:         ${custo_gen:.4f}")

    print(f"\nValidação:")
    print(f"  input tokens:  {CUSTO['val_input']}")
    print(f"  output tokens: {CUSTO['val_output']}")
    print(f"  custo:         ${custo_val:.4f}")

    print(f"\nTOTAL: ${custo_total:.4f}")
    print("===================\n")

def carregar_json(caminho):
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)

def validar_par(cliente, pergunta, resposta, contexto, modelo):
    prompt = f"""
        Verifique se a resposta está correta e totalmente suportada pelo texto.

        Texto:
        {contexto}

        Pergunta:
        {pergunta}

        Resposta:
        {resposta}

        Responda apenas:
        CORRETA ou INCORRETA
        """

    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Verifique se a resposta está correta com base no texto."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=10
    )

    usage = resp.usage

    CUSTO["val_input"] += usage.prompt_tokens
    CUSTO["val_output"] += usage.completion_tokens

    conteudo = resp.choices[0].message.content.strip().upper()
    return "CORRETA" in conteudo

def salvar_json(obj, caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def resposta_esta_no_texto(resposta, texto):
    t = normalizar(texto)
    r = normalizar(resposta)

    # heurística simples (baseline)
    return any(token in t for token in r.split() if len(token) > 4)

def sanitizar_texto(texto):

    limpo = "".join(
        c for c in texto
        if c in ("\n", "\t") or (not unicodedata.category(c).startswith("C"))
    )
    limpo = re.sub(r"\n{3,}", "\n\n", limpo)
    limpo = re.sub(r" {2,}", " ", limpo)
    return limpo.strip()


def truncar_texto(texto, max_chars = 3_000):
    texto = sanitizar_texto(texto)
    if len(texto) <= max_chars:
        return texto
    return texto[:max_chars] + " [... texto truncado]"


def calcular_max_chars_por_pub(n_pubs, budget_total = 12_000):
    por_pub = budget_total // max(1, n_pubs)
    return max(800, min(4_000, por_pub))

def distribuir_proporcional(pesos, total):

    n = len(pesos)
    if n == 0:
        return []
    soma = sum(pesos)
    if soma == 0:
        base = [total // n] * n
        resto = total - sum(base)
        for i in range(resto):
            base[i] += 1
        return base

    fracs = [p / soma * total for p in pesos]
    floors = [max(1, int(f)) for f in fracs]
    diff = total - sum(floors)

    residuos = sorted(
        range(n),
        key=lambda i: fracs[i] - int(fracs[i]),
        reverse=True,
    )
    for i in range(abs(diff)):
        idx = residuos[i % n]
        if diff > 0:
            floors[idx] += 1
        else:
            floors[idx] = max(1, floors[idx] - 1)

    return floors


def planejar_tarefas(dados, pares_por_chamada):
    tarefas = []

    for doc in dados:
        tarefas.append({
            "secao": doc["secao"],
            "data": doc["data"],
            "publicacoes": [doc],
            "n_pares": pares_por_chamada
        })

    return tarefas


def montar_prompt(tarefa, max_chars_override):
    n_pubs = len(tarefa["publicacoes"])
    max_chars = max_chars_override or calcular_max_chars_por_pub(n_pubs)

    blocos = []
    for i, pub in enumerate(tarefa["publicacoes"], 1):
        blocos.append(
            f"[Publicação {i}]\n"
            f"Tipo: {pub['tipo_documento']}\n"
            f"Número/Título: {pub.get('numero_documento', '')}\n"
            f"Texto: {truncar_texto(pub['texto_completo'], max_chars)}"
        )

    estilos_str = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(ESTILOS))

    return USER_TEMPLATE.format(
        n_pubs=n_pubs,
        secao=tarefa["secao"],
        data=tarefa["data"],
        edicao="N/A",
        blocos="\n\n".join(blocos),
        n_pares=tarefa["n_pares"],
        estilos=estilos_str,
    )

def chamar_api(cliente, tarefa, modelo):

    budget_steps = [None, 6000, 3000, 1500, 800]

    tentativa_global = 0
    budget_idx = 0

    while tentativa_global < MAX_RETRIES:
        tentativa_global += 1
        max_chars = budget_steps[min(budget_idx, len(budget_steps) - 1)]

        try:
            prompt = montar_prompt(tarefa, max_chars_override=max_chars)

            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,  # reduz custo
            )

            conteudo = resp.choices[0].message.content.strip()

            usage = resp.usage

            CUSTO["gen_input"] += usage.prompt_tokens
            CUSTO["gen_output"] += usage.completion_tokens

            # remove markdown
            if "```" in conteudo:
                linhas = [
                    l for l in conteudo.splitlines()
                    if not l.strip().startswith("```")
                ]
                conteudo = "\n".join(linhas).strip()

            # tenta extrair JSON
            inicio = conteudo.find("[")
            fim = conteudo.rfind("]")
            if inicio == -1 or fim == -1:
                raise ValueError("JSON não encontrado na resposta")

            conteudo = conteudo[inicio:fim + 1]

            pares = json.loads(conteudo)

            if not isinstance(pares, list):
                raise ValueError("Resposta não é lista")

            texto_base = " ".join(p["texto_completo"] for p in tarefa["publicacoes"])

            resultado = []

            for item in pares:
                if not isinstance(item, dict):
                    continue

                if "pergunta" not in item or "resposta" not in item:
                    continue

                pergunta = str(item["pergunta"]).strip()
                resposta = str(item["resposta"]).strip()

                if validar_par(cliente, pergunta, resposta, texto_base, modelo):
                    resultado.append({
                        "pergunta": pergunta,
                        "resposta": resposta,
                    })

            return resultado

        except Exception as e:
            # retry com backoff
            time.sleep(BACKOFF_BASE ** tentativa_global)

    return []

def processar_tarefa(args):
    idx, tarefa, cliente, modelo = args

    pares = chamar_api(cliente, tarefa, modelo)

    enriquecidos = [
        {
            "pergunta": p["pergunta"],
            "resposta": p["resposta"],
            "secao": tarefa["secao"],
            "fonte": tarefa["data"],
            "tipo_documento": tarefa["publicacoes"][0].get("tipo_documento", ""),
        }
        for p in pares
    ]

    return idx, tarefa, enriquecidos


def executar(dados, total_pares, modelo, pares_por_chamada,
             max_workers, caminho_checkpoint, caminho_saida):

    api_key = "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"#os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERRO] Defina a variável de ambiente OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    cliente = OpenAI(api_key=api_key)

    dataset = []
    tarefas_concluidas = set()

    # checkpoint
    if caminho_checkpoint and caminho_checkpoint.exists():
        checkpoint = carregar_json(caminho_checkpoint)
        dataset = checkpoint.get("dataset", [])
        tarefas_concluidas = set(checkpoint.get("tarefas_concluidas", []))
        print(f"Checkpoint carregado: {len(dataset)} pares já existentes.")

    # planejamento
    tarefas = planejar_tarefas(dados, pares_por_chamada)
    print(f"Total de tarefas planejadas: {len(tarefas)}")

    from collections import Counter
    dist_sec = Counter(t["secao"] for t in tarefas)
    dist_pares = Counter(t["secao"] for t in tarefas)

    print("\nDistribuição planejada de pares por seção:")
    for sec in sorted(dist_sec):
        print(f"  {sec}: {dist_sec[sec]} tarefas")
    print()

    tarefas_pendentes = [
        (i, t) for i, t in enumerate(tarefas)
        if i not in tarefas_concluidas
    ]

    print(f"Tarefas pendentes: {len(tarefas_pendentes)}")

    concluidas = 0
    total_tarefas = len(tarefas_pendentes)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:

        futuros = {
            pool.submit(processar_tarefa, (i, t, cliente, modelo)): (i, t)
            for i, t in tarefas_pendentes
        }

        for futuro in as_completed(futuros):
            try:
                idx, tarefa, pares_enriq = futuro.result()
            except Exception as exc:
                print(f"\n[ERRO inesperado] {exc}", file=sys.stderr)
                continue

            dataset.extend(pares_enriq)
            tarefas_concluidas.add(idx)
            concluidas += 1

            pct = (concluidas / total_tarefas * 100) if total_tarefas > 0 else 100

            print(
                f"\r[{concluidas}/{total_tarefas}] {pct:.1f}% | "
                f"Pares: {len(dataset)} | "
                f"{tarefa['secao']} — {tarefa['data']} (+{len(pares_enriq)})",
                end="",
                flush=True,
            )

            # checkpoint incremental
            if caminho_checkpoint and concluidas % 20 == 0:
                salvar_json(
                    {
                        "dataset": dataset,
                        "tarefas_concluidas": list(tarefas_concluidas)
                    },
                    caminho_checkpoint
                )

    print()

    # checkpoint final
    if caminho_checkpoint:
        salvar_json(
            {
                "dataset": dataset,
                "tarefas_concluidas": list(tarefas_concluidas)
            },
            caminho_checkpoint
        )

    # ajuste final
    if len(dataset) > total_pares:
        print(f"\n[INFO] {len(dataset)} pares gerados. Truncando para {total_pares}.")
        dataset = dataset[:total_pares]

    elif len(dataset) < total_pares:
        print(
            f"\n[AVISO] Apenas {len(dataset)} pares gerados (esperado: {total_pares}).",
            file=sys.stderr,
        )

    salvar_json(dataset, caminho_saida)

    print(f"\nDataset salvo em: {caminho_saida.resolve()}")
    print(f"Total de pares: {len(dataset)}")

    imprimir_custo()

def main():
    parser = argparse.ArgumentParser(
        description="Gera pares QA a partir do JSON consolidado do DODF via OpenAI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("resultado_saude2025.json"),
        help="JSON de entrada (saída de processar_dodf.py).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("dataset_qa_dodf.json"),
        help="Arquivo JSON de saída com os pares QA.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TOTAL_PARES,
        help=f"Total de pares a gerar (padrão: {TOTAL_PARES}).",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=MODELO_PADRAO,
        help=f"Modelo OpenAI a usar (padrão: {MODELO_PADRAO}).",
    )
    parser.add_argument(
        "--pares-por-chamada",
        type=int,
        default=PARES_POR_CHAMADA,
        dest="pares_por_chamada",
        help=f"Pares solicitados por chamada à API (padrão: {PARES_POR_CHAMADA}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        dest="max_workers",
        help=f"Chamadas paralelas à API (padrão: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Arquivo de checkpoint para retomar em caso de interrupção.",
    )
    args = parser.parse_args()

    if not args.entrada.exists():
        print(f"[ERRO] Arquivo de entrada não encontrado: {args.entrada}", file=sys.stderr)
        sys.exit(1)

    dados = carregar_json(args.entrada)


    limite = datetime(2025, 8, 1)

    dados = [
        doc for doc in dados
        if "data" in doc and datetime.fromisoformat(doc["data"]) >= limite
    ]

    print(f"Dados carregados: {len(dados)} dia(s).")

    executar(
        dados=dados,
        total_pares=args.total,
        modelo=args.modelo,
        pares_por_chamada=args.pares_por_chamada,
        max_workers=args.max_workers,
        caminho_checkpoint=args.checkpoint,
        caminho_saida=args.saida,
    )


if __name__ == "__main__":
    main()
