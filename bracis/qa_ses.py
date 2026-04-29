import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from openai import OpenAI, RateLimitError

TOTAL_MULTI = 10
MODELO = "gpt-4o-mini"
MAX_WORKERS = 1
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
TPM_LIMIT = 200_000
TPM_SOFT = int(TPM_LIMIT * 0.75)

CONTEXT_WINDOW = 128_000
OUTPUT_RESERVADO = 800
PROMPT_OVERHEAD = 600
MAX_TOKENS_INPUT = CONTEXT_WINDOW - OUTPUT_RESERVADO - PROMPT_OVERHEAD

MAX_TOKENS_DOC_MULTI = 2_000
MAX_DOCS_MULTI = 10

API_KEY=os.environ["OPENAI_API_KEY"]

CUSTO = {"gen_input": 0, "gen_output": 0, "val_input": 0, "val_output": 0}

tokens_usados = 0
janela_inicio = time.time()

_enc = tiktoken.encoding_for_model("gpt-4o-mini")


def contar_tokens(texto):
    return len(_enc.encode(texto))


def truncar_tokens(texto, max_tokens):
    texto = sanitizar_texto(texto)
    ids = _enc.encode(texto)
    if len(ids) <= max_tokens:
        return texto
    return _enc.decode(ids[:max_tokens]) + " [... texto truncado]"


SYSTEM_PROMPT = """\
Você é um especialista em saúde pública, com foco na análise de publicações do Diário Oficial do Distrito Federal (DODF). \
Sua tarefa é criar pares de PERGUNTA e RESPOSTA baseados exclusivamente no texto fornecido. As respostas devem ser precisas e fundamentadas \
no conteúdo da publicação — nunca invente informações ausentes no texto.
"""

USER_TEMPLATE_DIRECT = """\
Abaixo está UMA publicação do DODF:

{contexto}

LOCALIDADES ENCONTRADAS:
{lista_loc}

TÓPICOS ENCONTRADOS:
{topicos}

==============================
TAREFA
==============================

Gere exatamente {n_pares} pares de PERGUNTA e RESPOSTA com base no texto.

==============================
TIPO DE PERGUNTA
==============================

Cada pergunta DEVE conter explicitamente UMA localidade presente em LOCALIDADES ENCONTRADAS.

Se existir pelo menos um item em TÓPICOS ENCONTRADOS:
    - ENTÃO a pergunta DEVE estar contextualizada em pelo menos um desses tópicos
    - SENÃO ignore essa regra

==============================
REGRAS CRÍTICAS
==============================

Você deve gerar perguntas diretas contendo explicitamente uma localidade.

A) PROIBIÇÃO DE REFERÊNCIA DOCUMENTAL EXPLÍCITA

Perguntas NÃO podem conter estruturas nominais associadas a documentos, como:

- "do contrato"
- "do termo"
- "do valor"
- "do pregão"
- "da ata"

Também NÃO podem usar expressões como:

- "relacionado ao contrato"
- "previsto no contrato"
- "referente ao contrato"

*** Exemplos válidos ***

- "Qual aquisição foi realizada para a UBS do Gama?"
- "Qual é a situação sobre o fornecimento de insumos de enfermaria em Ceilândia?"
- "Quais serviços foram contratados para a UBS Santa Maria?"

*** Exemplos inválidos ***

- "Qual é a finalidade do contrato para a UPA de Ceilândia?"
    ERRO: contém "do contrato", contradiz a regra A.
- "Qual é o valor recebido para campanha de vacinação na localidade de Ceilândia?"
    ERRO: redundância "localidade de Ceilândia", contradizendo a regra C

IMPORTANTE: NÃO COLOQUE "localidade de..." ou "região de..." antes de mencionar explicitamente a região nas perguntas. Evite redundância e seja direto.

==============================
RESPOSTA
==============================

A resposta deve:

- estar explicitamente suportada pelo texto, podendo estar no início, meio ou final, conforme os exemplos abaixo
    "De acordo com o Contrato nº [X], publicado na Edição [Y] do DODF, de [data], ...",
    "Conforme o Termo Aditivo nº [X], publicado na Edição [Y] do DODF, de [data], ...",
    "... foi mencionado no Termo de Doação nº [X], publicado na Edição [Y] do DODF, de [data], ...",
- ser completa e natural

Exemplo válido:

- Para a pergunta "Qual unidade de saúde foi beneficiada com a aquisição de um novo lote de vacinas?", a resposta VÁLIDA é: "A UBS 12 da região do Gama recebeu um lote de vacinas contra a febre amarela em virtude do Termo de Doação nº 38/2025, publicado na Edição 05 do DODF, de 22/11/2025."

==============================
SAÍDA (JSON)
==============================

{{
  "data": [
    {{
      "pergunta": "...",
      "resposta": "..."
    }}
  ]
}}

Retorne APENAS o JSON.
Sem explicações.
"""

################################

USER_TEMPLATE_MULTI = """\
Abaixo existem diversas publicações do DODF:

{contexto}

LOCALIDADE-ALVO:
{lista_loc}

TÓPICOS ASSOCIADOS:
{topicos}

==============================
TAREFA
==============================

Você deve gerar pares de PERGUNTA e RESPOSTA baseados EXCLUSIVAMENTE no conteúdo acima.

A geração deve ser CONTROLADA pela LOCALIDADE-ALVO.

==============================
1) VALIDAÇÃO (OBRIGATÓRIA)
==============================

- Verifique se a LOCALIDADE-ALVO aparece no texto
- Se NÃO aparecer, retorne:

{{"data": []}}

- Verifique quais tópicos realmente aparecem no texto

Defina internamente:
- TOPICOS_VALIDOS = subconjunto de TÓPICOS que aparecem no texto

==============================
2) MODO DE GERAÇÃO
==============================

Se existem TOPICOS_VALIDOS, então

    Gere EXATAMENTE 1 par por tópico

    Para cada tópico T:
        - A pergunta DEVE envolver a LOCALIDADE-ALVO
        - A pergunta DEVE estar relacionada ao tópico T
        - A pergunta deve focar em ação, finalidade ou serviço

Se TOPICOS_VALIDOS é vazio, deve-se gerar EXATAMENTE 1 par

    - A pergunta DEVE envolver a LOCALIDADE-ALVO em todo o conjunto de publicações
    - A pergunta deve focar em ação, finalidade ou serviço presente no texto

==============================
3) REGRAS DAS PERGUNTAS
==============================

Todas as perguntas devem:

- Conter explicitamente a LOCALIDADE-ALVO
- Ser diretas e naturais
- Devem cobrir aspectos em comum em várias publicações
- Estar relacionadas à saúde pública
- NÃO conter:
    * "do contrato"
    * "do termo"
    * "do valor"
    * qualquer referência nominal direta a documento

- É PROIBIDO ELABORAR PERGUNTAS REDUNDANTES, como incluir "localidade de...". Por exemplo, "localidade de Ceilândia" ESTÁ ERRADO. Coloque apenas "Ceilândia".

- NÃO misturar múltiplas localidades

As perguntas devem focar em:

- ações realizadas
- serviços prestados
- aquisições
- finalidades institucionais
- atividades relacionadas à saúde pública

3.1. EXEMPLOS DE PERGUNTAS VÁLIDAS

- "Quais contratos abordam sobre os serviços hospitalares no DF?"
- "Quais empresas estão prestando serviços para a UBS de Santa Maria?"
- "Quais são as finalidades dos convênios associados à UBS da Asa Sul?"

==============================
4) RESPOSTAS
==============================

As respostas devem:
- Ser naturais e completas
- Conter referência explícita ao documento, variando-se a linguagem:
  Ex:
  "De acordo com os Contratos nº X, publicados nas Edições Y e Z do DODF, de [data] e [data], ..."
  "... foi mencionado no Termo de Doação nº [X], publicado na Edição [Y] do DODF, de [data], ...",

- Para múltiplas publicações:
  → incluir TODAS as referências relevantes

4.1. EXEMPLOS DE RESPOSTAS VÁLIDAS

- Para a pergunta "Quais contratos mencionam o SIA?", a resposta VÁLIDA é: "Os contratos nº 13/2024 e nº 75/2025, publicados nas Edições 01 e 03 do DODF de 12/05/2024 e 01/03/2025, respectivamente mencionam o SIA como centro de aquisição de equipamentos para as UBSs."
- Para a pergunta "Qual unidade de saúde foi beneficiada com a aquisição de um novo lote de vacinas?", a resposta VÁLIDA é: "A UBS 12 da região do Gama recebeu um lote de vacinas contra a febre amarela em virtude do Termo de Doação nº 38/2025, publicado na Edição 05 do DODF, de 22/11/2025."


==============================
5) PROIBIÇÕES
==============================

- Inventar localidades ou tópicos
- Inventar tópicos ou entidades
- Fazer perguntas genéricas ou vagas
- Fazer perguntas cuja resposta não está no texto
- Gerar respostas como "não há informação"
- Usar linguagem vaga como:
  - "o texto menciona"
  - "os documentos indicam"

==============================
7) FORMATO DE SAÍDA
==============================

{{
  "data": [
    {{
      "pergunta": "...",
      "resposta": "..."
    }}
  ]
}}

Retorne APENAS o JSON.
Sem explicações.
Sem texto adicional.
"""


def pergunta_invalida_tipoB(pergunta):
    p = pergunta.lower()

    # ❌ padrão nominal dependente
    if re.search(r"\b(do|da|dos|das)\s+(contrato|termo|preg[aã]o|ata|valor)\b", p):
        return True

    # ❌ também pega variações tipo "para o contrato"
    if re.search(r"\b(para|no|na)\s+(o|a)\s+(contrato|termo|preg[aã]o|ata)\b", p):
        return True

    return False

def pergunta_mal_formada(pergunta):
    return (
        pergunta.strip().endswith("?") is False or
        len(pergunta.split()) < 6
    )

def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_json(obj, caminho):
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def sanitizar_texto(texto):
    limpo = "".join(
        c for c in texto
        if c in ("\n", "\t") or not unicodedata.category(c).startswith("C")
    )
    limpo = re.sub(r"\n{3,}", "\n\n", limpo)
    return re.sub(r" {2,}", " ", limpo).strip()


def normalizar_item(item):
    return {k.strip().lower().replace('"', ''): v for k, v in item.items()}


def resposta_esta_no_texto(resposta, texto):
    t = normalizar(texto)
    r = normalizar(resposta)
    tokens = [tok for tok in r.split() if len(tok) > 4]
    if not tokens:
        return False
    matches = sum(1 for tok in tokens if tok in t)
    return matches / len(tokens) >= 0.3


def controlar_tpm(tokens_previstos):
    global tokens_usados, janela_inicio
    agora = time.time()
    if agora - janela_inicio >= 60:
        tokens_usados = 0
        janela_inicio = agora
    if tokens_usados + tokens_previstos > TPM_SOFT:
        espera = 60 - (agora - janela_inicio)
        if espera > 0:
            print(f"[TPM WAIT] {espera:.2f}s")
            time.sleep(espera)
        tokens_usados = 0
        janela_inicio = time.time()


def validar_par(cliente, pergunta, resposta, contexto):
    prompt = f"""Verifique se a resposta está correta e totalmente suportada pelo texto.

Texto:
{contexto}

Pergunta:
{pergunta}

Resposta:
{resposta}

Responda apenas: CORRETA ou INCORRETA"""

    resp = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": "Verifique se a resposta está correta com base no texto."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=10,
    )
    CUSTO["val_input"] += resp.usage.prompt_tokens
    CUSTO["val_output"] += resp.usage.completion_tokens
    return "CORRETA" in resp.choices[0].message.content.strip().upper()

def montar_prompt_direct(tarefa, max_tokens_override=None):
    max_tokens_doc = max_tokens_override or MAX_TOKENS_DOC_MULTI

    docs = tarefa["publicacoes"]

    if docs and "publicacoes" in docs[0]:
        docs = [p for d in docs for p in d.get("publicacoes", [])]

    if not docs:
        return "", []

    doc = docs[0]

    localidades = set()
    keywords = set()
    docs_no_bloco = []
    partes = []
    tokens_acumulados = 0

    header = (
        f"[FONTE]\nEdição {doc.get('edicao')}, publicada em {doc.get('data')}, "
        f"{doc.get('secao')}, {doc.get('referencia')}\n\n"
    )

    tokens_header = contar_tokens(header)

    tokens_disponiveis = min(
        max_tokens_doc,
        MAX_TOKENS_INPUT - tokens_acumulados - tokens_header
    )

    if tokens_disponiveis > 0:
        txt = truncar_tokens(doc.get("texto_completo", ""), tokens_disponiveis)
        parte = header + txt

        partes.append(parte)
        docs_no_bloco.append(doc)
        tokens_acumulados += contar_tokens(parte)

    for loc in (doc.get("localidades") or []):
        localidades.add(loc)

    for loc in (doc.get("tipo_unidade") or []):
        localidades.add(loc)

    for key in (doc.get("topicos") or []):
        keywords.add(key)

    for key in (doc.get("programas") or []):
        keywords.add(key)

    n_pares = tarefa.get("n_pares", 1)

    prompt = USER_TEMPLATE_DIRECT.format(
        contexto="\n\n".join(partes),
        lista_loc=", ".join(sorted(localidades)) if localidades else "N/A",
        topicos=", ".join(sorted(keywords)) if keywords else "N/A",
        n_pares=n_pares
    )

    return prompt, docs_no_bloco


def montar_prompt_multi(tarefa, max_tokens_override=None):
    max_tokens_doc = max_tokens_override or MAX_TOKENS_DOC_MULTI

    entidade = tarefa.get("entidade")
    topicos = tarefa.get("topicos", [])
    docs = tarefa.get("publicacoes", [])

    modo_topico = len(topicos) > 0

    docs_no_bloco = []
    partes = []
    tokens_acumulados = 0

    for d in docs:
        header = (
            f"[FONTE]\nEdição {d.get('edicao')}, publicada em {d.get('data')}, "
            f"{d.get('secao')}, {d.get('referencia')}\n\n"
        )

        tokens_header = contar_tokens(header)

        tokens_disponiveis = min(
            max_tokens_doc,
            MAX_TOKENS_INPUT - tokens_acumulados - tokens_header
        )

        if tokens_disponiveis <= 0:
            break

        txt = truncar_tokens(d.get("texto_completo", ""), tokens_disponiveis)

        parte = header + txt
        partes.append(parte)
        docs_no_bloco.append(d)

        tokens_acumulados += contar_tokens(parte)

    prompt = USER_TEMPLATE_MULTI.format(
        contexto="\n\n".join(partes),
        lista_loc=entidade if entidade else "N/A",
        topicos=", ".join(sorted(topicos)) if topicos else "N/A",
        n_pares=len(topicos) if modo_topico else 1,
        modo_topico="SIM" if modo_topico else "NAO"
    )

    return prompt, docs_no_bloco

def construir_indice_localidades(dados_flat):
    indice = defaultdict(lambda: {
        "docs": [],
        "topicos": set()
    })

    for d in dados_flat:
        localidades = d.get("localidades") or []
        topicos = d.get("topicos") or []

        for loc in localidades:
            entry = indice[loc]

            entry["docs"].append(d)

            for t in topicos:
                entry["topicos"].add(t)

    return indice

def construir_blocos_por_localidade(docs_multi, max_tokens_bloco, max_docs_por_bloco=None):
    blocos = []

    for grupo in docs_multi:
        docs = grupo["publicacoes"]

        bloco = []
        tokens_bloco = 0

        for d in docs:
            header = (
                f"Edição {d['edicao']} — {d['data_edicao']} — "
                f"{d['secao']} — {d.get('referencia','')}\n"
            )

            t = contar_tokens(header + d.get("texto_completo", ""))

            if bloco and (
                tokens_bloco + t > max_tokens_bloco or
                (max_docs_por_bloco and len(bloco) >= max_docs_por_bloco)
            ):
                blocos.append({
                    "entidade": grupo["entidade"],
                    "topicos": grupo["topicos"],
                    "docs": bloco
                })
                bloco = []
                tokens_bloco = 0

            bloco.append(d)
            tokens_bloco += t

        if bloco:
            blocos.append({
                "entidade": grupo["entidade"],
                "topicos": grupo["topicos"],
                "docs": bloco
            })

    return blocos


def _textos_doc(d):
    return normalizar(d.get("texto_completo", ""))


def _par_e_direto(pergunta, resposta, docs_no_bloco):
    """Retorna True se a resposta está concentrada em apenas um dos docs do bloco."""
    scores = []
    for d in docs_no_bloco:
        t = _textos_doc(d)
        tokens_resp = [tok for tok in normalizar(resposta).split() if len(tok) > 4]
        if not tokens_resp:
            scores.append(0.0)
            continue
        matches = sum(1 for tok in tokens_resp if tok in t)
        scores.append(matches / len(tokens_resp))
    if not scores:
        return False
    melhor = max(scores)
    scores_sig = [s for s in scores if s > 0.1]
    return melhor >= 0.5 and len(scores_sig) == 1

def extrair_pares(conteudo):
    try:
        parsed = json.loads(conteudo)
    except Exception:
        return []

    if isinstance(parsed, dict):
        # caso padrão esperado: {"data": [...]}
        if "data" in parsed and isinstance(parsed["data"], list):
            return parsed["data"]

        # fallback: dict direto com pergunta/resposta
        if "pergunta" in parsed and "resposta" in parsed:
            return [parsed]

        # fallback genérico
        for v in parsed.values():
            if isinstance(v, list):
                return v

        return []

    if isinstance(parsed, list):
        return parsed

    return []

def chamar_api(cliente, tarefa, max_docs_por_bloco=None):
    global tokens_usados

    docs = tarefa.get("publicacoes", [])
    if docs and "publicacoes" in docs[0]:
        docs = [p for d in docs for p in d.get("publicacoes", [])]

    if not docs:
        return []

    docs = sorted(docs, key=lambda d: d.get("data_edicao", ""))
    resultado = []

    entidade = tarefa.get("entidade")
    topicos = tarefa.get("topicos", [])

    max_tokens_bloco = MAX_TOKENS_INPUT // 2
    budget_steps = [None, max_tokens_bloco // 2, max_tokens_bloco // 4, 1_000]

    grupo = {
        "entidade": tarefa["entidade"],
        "topicos": tarefa.get("topicos", []),
        "publicacoes": tarefa["publicacoes"]
    }

    for bloco_info in construir_blocos_por_localidade([grupo], max_tokens_bloco, max_docs_por_bloco):
        bloco = bloco_info["docs"]

        tarefa_bloco = {
            **tarefa,
            "publicacoes": bloco,
            "entidade": bloco_info["entidade"],
            "topicos": bloco_info["topicos"]
        }
        if not bloco:
            continue

        pares_candidatos = []

        # 🔧 modo baseado na tarefa (não no tamanho)
        modo_direct = not tarefa.get("multi", False)

        for tentativa in range(MAX_RETRIES):
            try:
                max_tokens_doc = budget_steps[min(tentativa, len(budget_steps) - 1)]

                if modo_direct:
                    prompt, docs_no_bloco = montar_prompt_direct(tarefa_bloco, max_tokens_doc)
                else:
                    prompt, docs_no_bloco = montar_prompt_multi(tarefa_bloco, max_tokens_doc)

                tokens_prompt = contar_tokens(prompt)
                controlar_tpm(tokens_prompt + OUTPUT_RESERVADO)
                time.sleep(random.uniform(0.3, 0.6))

                resp = cliente.chat.completions.create(
                    model=MODELO,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=OUTPUT_RESERVADO,
                    response_format={"type": "json_object"},
                )

                if resp.usage:
                    tokens_usados += resp.usage.prompt_tokens + resp.usage.completion_tokens
                    CUSTO["gen_input"] += resp.usage.prompt_tokens
                    CUSTO["gen_output"] += resp.usage.completion_tokens

                pares_candidatos = extrair_pares(resp.choices[0].message.content.strip())

                if isinstance(pares_candidatos, list) and pares_candidatos:
                    break

            except RateLimitError as e:
                time.sleep(2.0)

            except Exception as e:
                print("[ERRO]", str(e))
                time.sleep(BACKOFF_BASE ** tentativa)

        if not pares_candidatos:
            continue

        texto_base = " ".join(d.get("texto_completo", "") for d in bloco)

        texto_publicacao = "\n\n".join(
            f"Edição {d.get('edicao')} — {d.get('data_edicao')} — {d.get('secao')} — {d.get('referencia','')}\n"
            f"{d.get('texto_completo','')}"
            for d in bloco
        )

        for item in pares_candidatos:
            if not isinstance(item, dict):
                continue

            item = normalizar_item(item)

            pergunta = str(item.get("question") or item.get("pergunta") or "").strip()
            resposta = str(item.get("answer") or item.get("resposta") or "").strip()

            if entidade:
                p_norm = normalizar(pergunta)
                if normalizar(entidade) not in p_norm:
                    continue

            if pergunta_invalida_tipoB(pergunta):
                continue

            if not pergunta or not resposta:
                continue

            if not resposta_esta_no_texto(resposta, texto_base):
                continue

            if not validar_par(cliente, pergunta, resposta, texto_base):
                continue

            resultado.append({
                "pergunta": pergunta,
                "resposta": resposta,
                "texto_publicacao": texto_publicacao,
            })

        time.sleep(0.5)

    return resultado

def processar_tarefa(args):
    idx, tarefa, cliente, max_docs_por_bloco = args
    pares = chamar_api(cliente, tarefa, max_docs_por_bloco)
    enriquecidos = [
        {
            "pergunta": p["pergunta"],
            "resposta": p["resposta"],
            "secao": tarefa["secao"],
            "fonte": tarefa["data"],
            "tipo_documento": tarefa["publicacoes"][0].get("tipo_documento", ""),
            "texto_publicacao": p.get("texto_publicacao", ""),
        }
        for p in pares
    ]
    return idx, tarefa, enriquecidos


def planejar_tarefas(dados, pares_por_chamada):
    return [
        {
            "multi": True,
            "entidade": doc.get("entidade"),
            "secao": "MULTI",
            "data": "MULTI",
            "publicacoes": doc.get("publicacoes", []),
            "n_pares": pares_por_chamada,
        }
        for doc in dados
    ]

def executar(dados, total_pares, pares_por_chamada, caminho_checkpoint, caminho_saida, max_docs_por_bloco=None):
    cliente = OpenAI(api_key=API_KEY)

    dataset = []
    tarefas_concluidas = set()

    # 🔢 Contadores globais
    total_gerados = 0
    total_aceitos = 0

    if caminho_checkpoint and os.path.exists(caminho_checkpoint):
        checkpoint = carregar_json(caminho_checkpoint)
        dataset = checkpoint.get("dataset", [])
        tarefas_concluidas = set(checkpoint.get("tarefas_concluidas", []))
        total_aceitos = len(dataset)
        print(f"Checkpoint carregado: {total_aceitos} pares.")

    tarefas = planejar_tarefas(dados, pares_por_chamada)
    tarefas_pendentes = [(i, t) for i, t in enumerate(tarefas) if i not in tarefas_concluidas]

    print(f"Total de tarefas: {len(tarefas)} | Pendentes: {len(tarefas_pendentes)}")

    concluidas = 0

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 2)) as pool:
        futuros = {
            pool.submit(processar_tarefa, (i, t, cliente, max_docs_por_bloco)): (i, t)
            for i, t in tarefas_pendentes
        }

        for futuro in as_completed(futuros):
            if total_aceitos >= total_pares:
                print("\n[INFO] Limite de pares atingido. Encerrando...")
                break

            try:
                idx, tarefa, pares_enriq = futuro.result()
            except Exception as exc:
                print(f"[ERRO] {exc}", file=sys.stderr)
                continue

            # 🔢 contabiliza TUDO que veio da API (antes de cortar)
            gerados = len(pares_enriq)
            total_gerados += gerados

            aceitos = 0

            if pares_enriq:
                restante = total_pares - total_aceitos
                if restante > 0:
                    selecionados = pares_enriq[:restante]
                    dataset.extend(selecionados)
                    aceitos = len(selecionados)
                    total_aceitos += aceitos
                    tarefas_concluidas.add(idx)

            concluidas += 1

            print(
                f"\r[{concluidas}/{len(tarefas_pendentes)}] "
                f"Gerados: {total_gerados} | "
                f"Aceitos: {total_aceitos} | "
                f"Tarefa(+{gerados}→{aceitos})",
                end="",
                flush=True,
            )

            if caminho_checkpoint and concluidas % 20 == 0:
                salvar_json(
                    {
                        "dataset": dataset,
                        "tarefas_concluidas": list(tarefas_concluidas),
                    },
                    caminho_checkpoint,
                )

        for f in futuros:
            f.cancel()

    print()

    if caminho_checkpoint:
        salvar_json(
            {
                "dataset": dataset,
                "tarefas_concluidas": list(tarefas_concluidas),
            },
            caminho_checkpoint,
        )

    salvar_json(dataset[:total_pares], caminho_saida)

    preco_input = 0.00015 / 1000
    preco_output = 0.0006 / 1000

    custo_gen = CUSTO["gen_input"] * preco_input + CUSTO["gen_output"] * preco_output
    custo_val = CUSTO["val_input"] * preco_input + CUSTO["val_output"] * preco_output

    print(f"\nDataset salvo: {caminho_saida}")
    print(f"Pares aceitos: {total_aceitos}")
    print(f"Pares gerados (antes de filtro): {total_gerados}")
    print(f"Custo geração: ${custo_gen:.4f} | Validação: ${custo_val:.4f} | Total: ${custo_gen + custo_val:.4f}")


def flatten_dados(dados):
    if isinstance(dados, list):
        return dados
    return [doc for info in dados.values() for doc in info.get("publicacoes", [])]


def construir_indice_unidades(dados_flat):
    indice = defaultdict(list)
    for doc in dados_flat:
        for loc in (doc.get("localidades") or []):
            indice[normalizar(loc)].append(doc.get("id"))
    return dict(indice)


def construir_indice_localidades(dados_flat):
    indice = defaultdict(lambda: {
        "docs": [],
        "topicos": set()
    })

    for d in dados_flat:
        localidades = d.get("localidades") or []
        topicos = d.get("topicos") or []

        for loc in localidades:
            entry = indice[loc]

            entry["docs"].append(d)

            for t in topicos:
                entry["topicos"].add(t)

    return indice


dados = carregar_json("data/corpus_dodf_2025.json")
dados_flat = flatten_dados(dados)
indice = construir_indice_localidades(dados_flat)

docs_multi = []
for loc, data in indice.items():
    if len(data["docs"]) < 2:
        continue

    docs_multi.append({
        "id": f"multi_{loc}",
        "entidade": loc,
        "topicos": list(data["topicos"]),
        "multi": True,
        "publicacoes": data["docs"]
    })

print(f"Total entidades: {len(indice)}")
print(f"Entidades multi: {len([k for k, v in indice.items() if len(v) >= 2])}")
print(f"Budget input disponível: {MAX_TOKENS_INPUT:,} tokens")

#executar(
#    dados=docs_multi,
#    total_pares=1000,
#    pares_por_chamada=2,
#    caminho_checkpoint="checkpoint_qa_single.json",
#    caminho_saida="data/dataset_qa_single.json",
#    max_docs_por_bloco=1
#)

executar(
    dados=docs_multi,
    total_pares=1000,
    pares_por_chamada=2,
    caminho_checkpoint="checkpoint_qa_multi5.json",
    caminho_saida="data/dataset_qa_multi5.json",
    max_docs_por_bloco=5
)

executar(
    dados=docs_multi,
    total_pares=100,
    pares_por_chamada=2,
    caminho_checkpoint="checkpoint_qa_multi10.json",
    caminho_saida="data/dataset_qa_multi10.json",
    max_docs_por_bloco=5
)
