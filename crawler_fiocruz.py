from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

chromedriver = Service(ChromeDriverManager().install())
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.page_load_strategy = "eager"
driver = webdriver.Chrome(service=chromedriver, options=options)

news_data = []

page = 1
page_init = page

while page <= 5:

    print(f'Page: {page}')

    NEWS_URL = "https://fiocruz.br/noticias?search_api_fulltext=Bras%C3%ADlia&field_editora=All&field_taxonomia_doencas=All&field_unidade_curso=All&page="+str(page)
    driver.get(NEWS_URL)

    time.sleep(0.5)

    wait = WebDriverWait(driver, 10)

    container = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="block-portal-fiocruz-conteudodapaginaprincipal"]/div/div')))

    links = container.find_elements(By.TAG_NAME, "a")

    news_links = []

    for link in links:
        a = link.get_attribute('href')
        if "/noticia/" in a:
            news_links.append(a)

    news_links = list(set(news_links))

    print(news_links[:3])


    for link in news_links:
        driver.get(link)
        wait = WebDriverWait(driver, 10)

        #print(link)

        try:
            title = wait.until(
                EC.presence_of_element_located(
                   (By.XPATH, "/html/body/div[2]/main/div/div[2]/div[1]/h1")
                )
            ).text.strip()

            print(title)

            container = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, '//div[contains(@class,"field--name-body") and contains(@class,"field__item")]//p')
                )
            )

            paragraphs = container.find_elements(By.XPATH, ".//p")

            if not paragraphs:
                paragraphs = driver.find_elements(By.XPATH, "//article//p")

            texts = [p.text.strip() for p in paragraphs if p.text.strip()]
            content = "\n\n".join(texts)

            if len(content) > 10:
                news_data.append({
                    "title": title,
                    "content": content,
                    "url": link
                })
                #print(f"Coletado ({len(content)} chars)")

            else:
                print("Conteudo vazio ou insuficiente")

        except Exception as e:
            print(f"Erro ao coletar {link}: {e}")
            #driver.get(NEWS_URL)

    #for news in news_data:
    #    print("\nTítulo:", news["title"])
    #    print("Conteúdo:", news["content"][:500], "...")
    #    print("Link:", news["url"])

    page+=1

driver.quit()

df = pd.DataFrame(news_data)

df.to_csv(f"fiocruz_noticias_{page_init}_{page}.csv", index=False, encoding="utf-8")
