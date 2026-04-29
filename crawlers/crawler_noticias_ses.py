import re
import unicodedata
import pandas as pd
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TARGET_NEWS = 5000
MAX_PAGES = 400

# --------------------------------------------------
# NORMALIZAÇÃO
# --------------------------------------------------

def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# LISTAS
# --------------------------------------------------

LOCALIDADES_DF = [
    "plano piloto",
    "asa sul",
    "regiao sul",
    "regiao norte",
    "regiao centro oeste",
    "regiao leste",
    "regiao oeste",
    "região sudoeste",
    "regiao central",
    "nucleo rural",
    "capao seco",
    "barreiro",
    "asa norte",
    "vila planalto",
    "noroeste",
    "gama",
    "taguatinga",
    "brazlandia",
    "sobradinho",
    "planaltina",
    "paranoa",
    "setor habitacional",
    "nucleo bandeirante",
    "ceilandia",
    "guara",
    "cruzeiro",
    "samambaia",
    "santa maria",
    "cafe sem troco",
    "paddf",
    "santa barbara",
    "tororo",
    "sao sebastiao",
    "recanto das emas",
    "lago sul",
    "ponte alta",
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
    "ubs",
    "unidade basica de saude",
    "unidade basica",
    "hospital",
    "upa",
    "unidade de pronto atendimento"
]


def build_regex(pattern_list):
    escaped = [re.escape(normalize(p)) for p in pattern_list]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


REGEX_LOCAL = build_regex(LOCALIDADES_DF)
REGEX_UNIDADE = build_regex(TIPOS_UNIDADE)


# --------------------------------------------------
# SELENIUM
# --------------------------------------------------

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--disable-extensions")
options.add_argument("--disable-infobars")
options.add_argument("--disable-notifications")


driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver, 10)


# --------------------------------------------------
# URL
# --------------------------------------------------

NEWS_URL = (
    "https://saude.df.gov.br/noticias?"
    "p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L"
    "&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L_cur="
)


# --------------------------------------------------
# COLETA
# --------------------------------------------------

rows = []
visited = set()

for page in range(1, MAX_PAGES + 1):

    if page % 20 == 0:
        driver.quit()
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    if len(rows) >= TARGET_NEWS:
        break

    print(f"\nPágina {page} | Coletadas: {len(rows)}")

    driver.get(NEWS_URL + str(page))

    try:
        container = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                '//*[@id="portlet_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L"]'
            ))
        )
    except:
        print("Falha ao carregar página")
        continue

    links = container.find_elements(
        By.XPATH, ".//a[starts-with(@href, 'https://saude.df.gov.br/w/')]"
    )

    hrefs = sorted({
        l.get_attribute("href") for l in links if l.get_attribute("href")
    })

    for href in hrefs:

        if href in visited:
            continue

        visited.add(href)

        if len(rows) >= TARGET_NEWS:
            break

        try:
            driver.get(href)

            title = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[starts-with(@id,"fragment-")]//h3')
                )
            ).text.strip()

            print(title)
            paragraphs = wait.until(
                EC.presence_of_all_elements_located((
                    By.XPATH,
                    '//*[starts-with(@id,"fragment-")]//p'
                ))
            )

            content = "\n".join(
                p.text.strip() for p in paragraphs if p.text.strip()
            )

            if not content:
                continue

            full_text = normalize(title + " " + content)

            # ----------------------------
            # FILTRO COM REGEX
            # ----------------------------

            locais = REGEX_LOCAL.findall(full_text)
            unidades = REGEX_UNIDADE.findall(full_text)

            if not locais or not unidades:
                continue

            rows.append({
                "title": title,
                "content": content,
                "url": href,
                "localidades": ";".join(set(locais)),
                "tipos_unidade": ";".join(set(unidades))
            })

            print(f"OK ({len(rows)})")

        except Exception as e:
            print("Erro:", e)
            continue


# --------------------------------------------------
# FINALIZAÇÃO
# --------------------------------------------------

driver.quit()

df = pd.DataFrame(rows)
df.to_csv("noticias_ses_5000.csv", index=False, encoding="utf-8")

print(f"\nTotal coletado: {len(df)}")
