from urllib.request import urlopen, Request

## from Inagenet

DIR="/Workspace-Github/fruit_classification/images/"
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
url_list = []

def downloader(image_url, file):
    print(image_url)
    if image_url.endswith('.png'):
        full_file_name = str(file) + '.png'
    else:
        full_file_name = str(file) + '.jpg'
    try:
        req = Request(image_url, headers=header)
        raw_img = urlopen(req, timeout=5).read()
        f = open(full_file_name, 'wb')
        f.write(raw_img)
        f.close()
    except:
        print('Can not download: ', image_url)

            
for i in ['lemon', 'orange', 'tangerine', 'grapefruit']:
    count = 0
    urls = open(DIR + i + '.txt').read().split('\n')
    for url in urls:
        file = DIR + i + '/' + str(count)
        downloader(url, file)
        count += 1
    