def c(link):
    import requests 
    import pandas as pd 
    from bs4 import BeautifulSoup 
    # link for extract html data 
    def getdata(url): 
        r = requests.get(url) 
        return r.text 

    htmldata = getdata(link) 
    soup = BeautifulSoup(htmldata, 'html.parser') 
    data = '' 
    fdata = ''
    for data in soup.find_all("p"): 
        fdata = fdata +" "+ data.get_text()
    return fdata
