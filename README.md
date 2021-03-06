# Spiders

## Code

```
$ scrapy startproject spiders
$ cd spiders
$ scrapy genspider npg www.npg.org.uk
$ scrapy runspider --nolog npg.py
```

```
$ scrapy runspider --nolog npg.py
# the above command can be replaced with the command below:
$ scrapy crawl npg
```

```
$ pip3 install selenium
$ brew cask install chromedriver
$ chromedriver --version
```

```
from selenium import webdriver

# initialize the driver
driver = webdriver.Chrome()

# open provided link in a browser window using the driver
driver.get("https://google.com")
```

## Steps

<p float="left">
	<img src="./pix/spiders.png" width=800 />
</p>

## Dataset

<p float="left">
	<img src="./pix/image_downloader.png" width=800 />
</p>


## References
* https://scrapy.org/
* https://docs.scrapy.org/en/latest/topics/item-pipeline.html
* https://docs.scrapy.org/en/latest/topics/feed-exports.html#topics-feed-exports
* https://stackoverflow.com/questions/43922562/scrapy-how-to-use-items-in-spider-and-how-to-send-items-to-pipelines
* https://selenium-python.readthedocs.io/getting-started.html
* https://www.npg.org.uk/research/programmes/making-art-in-tudor-britain/matbsearch
* https://www.tensorflow.org/guide/eager#computing_gradients
