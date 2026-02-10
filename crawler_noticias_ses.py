from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import time
import unicodedata
import re

# --------------------------------------------------
# Normalização de texto
# --------------------------------------------------

def normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------------------------------------
# Configurações
# --------------------------------------------------

UBSES = [
    "UBS 1 de Águas Claras", "UBS 2 de Águas Claras",
    "UBS 1 da Asa Norte", "UBS 2 da Asa Norte",
    "UBS 1 da Asa Sul",
    "UBS 1 de Brazlândia", "UBS 2 de Brazlândia",
    "UBS 1 do Gama", "UBS 2 do Gama",
    "UBS 1 do Cruzeiro"
]

NEWS_URL = (
    "https://saude.df.gov.br/noticias?"
    "p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L"
    "&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view"
    "&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L_delta=10"
    "&p_r_p_resetCur=false"
    "&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L_cur="
)

MAX_PAGES = 400

# --------------------------------------------------
# Selenium setup
# --------------------------------------------------

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver, 10)

# --------------------------------------------------
# Coleta
# --------------------------------------------------

rows = []

page_init = 1
page_end = 100

for page in range(page_init, page_end + 1):
    print(f"Página {page}")
    driver.get(NEWS_URL + str(page))
    wait = WebDriverWait(driver, 5)

    try:
        container = wait.until(
            EC.presence_of_element_located((
                By.XPATH,
                '//*[@id="portlet_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L"]/div'
            ))
        )
    except Exception:
        print("Container não encontrado, pulando página.")
        continue

    links = container.find_elements(
        By.XPATH, ".//a[starts-with(@href, 'https://saude.df.gov.br/w/')]"
    )

    hrefs = sorted({
        link.get_attribute("href")
        for link in links
        if link.get_attribute("href")
    })

    for href in hrefs:
        print("Coletando:", href)
        try:
            driver.get(href)
            wait = WebDriverWait(driver, 10)

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

            content = "\n\n".join(
                p.text.strip() for p in paragraphs if p.text.strip()
            )

            if not content:
                print("Conteúdo vazio, ignorando.")
                continue

            texto_completo = f"{title}\n\n{content}"

            matched_ubs = [
                ubs for ubs in UBSES
                if normalize(ubs) in normalize(texto_completo)
            ]

            rows.append({
                "title": title,
                "content": content,
                "url": href,
                "matched_ubs": "; ".join(matched_ubs)
            })

            print(f"OK ({len(content)} chars)")

        except Exception as e:
            print(f"Erro ao coletar {href}: {e}")
            continue

# --------------------------------------------------
# Finalização
# --------------------------------------------------

driver.quit()

df = pd.DataFrame(rows)
df.to_csv(f"noticias_ses_{page_init}_{page_end}.csv", index=False, encoding="utf-8")

print("CSV gerado: noticias_ses.csv")
