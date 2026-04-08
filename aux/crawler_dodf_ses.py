from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://dodf.df.gov.br/dodf/jornal/diario?data=1751857200"
driver.get(url)

driver.implicitly_wait(10)



container = driver.find_element(By.XPATH, "/html/body/div[12]/section/div/div/div[2]/main/div/div/ul")
#container = driver.find_element(By.XPATH, "/html/body/div[12]/section/div/div/div[2]")

links = container.find_elements(By.TAG_NAME, "a")
urls = [link.get_attribute("href") for link in links if link.get_attribute("href")]

for url in urls:
    print(url)

driver.quit()
