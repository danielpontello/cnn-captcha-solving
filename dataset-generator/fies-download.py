import urllib.request

url = "http://sisfiesaluno.mec.gov.br/principal/captcha"

for i in range(500):
    try:
        print("Baixando captcha " + str(i) + "...")
        filename = "../dataset/test/" + str(i).zfill(5) + ".png"
        urllib.request.urlretrieve(url, filename)
    except:
        print("Erro!")
    