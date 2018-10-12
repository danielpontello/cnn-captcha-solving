import urllib.request

url = "http://sisfiesaluno.mec.gov.br/principal/captcha"

for i in range(1000):
    try:
        print("Baixando captcha " + str(i) + "...")
        filename = "raw-images/" + str(i).zfill(5) + ".png"
        urllib.request.urlretrieve(url, filename)
    except:
        print("Erro!")
    