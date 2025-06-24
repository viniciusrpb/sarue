from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from collections import defaultdict
import json
import time

ubses = [
    "UBS 1 de Águas Claras", "UBS 2 de Águas Claras", "UBS 1 da Asa Norte", "UBS 2 da Asa Norte", "UBS 1 da Asa Sul", "UBS 1 de Brazlândia", "UBS 2 de Brazlândia",
    "UBS 1 do Gama", "UBS 2 do Gama", "UBS 1 do Cruzeiro"
]

chromedriver = Service(ChromeDriverManager().install())
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=chromedriver, options=options)

NEWS_URL = "https://saude.df.gov.br/noticias?p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L_delta=10&p_r_p_resetCur=false&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L_cur="

wait = WebDriverWait(driver, 10)

page = 1

MAX_PAGES = 20

resultados = defaultdict(list)

while page <= MAX_PAGES:

    print(f'Page: {page}')

    PAGE_URL = NEWS_URL+str(page)

    driver.get(PAGE_URL)

    wait = WebDriverWait(driver, 5)

    container_xpath = '//*[@id="portlet_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_Cziz3oWq1x3L"]/div'
    container = wait.until(EC.presence_of_element_located((By.XPATH, container_xpath)))

    noticia_links = container.find_elements(By.XPATH, ".//a[starts-with(@href, 'https://saude.df.gov.br/w/')]")

    hrefs = sorted(set(link.get_attribute("href") for link in noticia_links if link.get_attribute("href")))

    print(hrefs)

    n = 1

    for href in hrefs:
        try:
            print(f"Link {n}")
            print(href)
            driver.get(href)
            time.sleep(2)

            titulo_el = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="fragment-0-mbiu"]/div/div/div/h3')))
            titulo = titulo_el.text.strip()

            corpo_el = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="fragment-0-mbiu"]/div')))
            corpo = corpo_el.text.strip()

            texto_completo = titulo + " " + corpo

            #print('\n\n\n\nTexto completo:')
            #print(texto_completo)
            #print('\n\n\n\n')

            for ubs in ubses:
                if normalize(ubs) in normalize(texto_completo):
                    resultados[ubs].append({
                        "titulo": titulo,
                        "link": href,
                        "trecho": corpo[:700]
                    })
                    print(f"Encontrado: {ubs} → {titulo}")
                    break

        except Exception as e:
            print(f"Erro em {href}: {e}")
            continue

        n+=1

    page+=1

driver.quit()

f = open("noticias.json", "w", encoding="utf-8")
json.dump(resultados_por_ubs, f, ensure_ascii=False, indent=2)
