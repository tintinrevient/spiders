import json
import csv
import re
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
from time import time
import pyautogui
import pyperclip

def convert_file(infile, outfile):

    data = []

    with open(infile, encoding='utf-8') as fh:
        # read all the lines
        lines = fh.readlines()

        for line in lines:
            # convert str to dict
            line = json.loads(line)

            name = line['name'].replace(";", ",").replace("\"", "")
            desp = line['desp']
            url = line['url']

            # regex of the pattern '(1890-1967), Sculptor. 8 Portraits'
            match_obj = re.match(r'\s*\((.*)-(.*)\),\s+(.*)\.\s+(.*)\s*', desp, re.M|re.I)

            # if matched
            if match_obj:
                # convert str to int
                try:
                    start_year = int(match_obj[1])
                    end_year = int(match_obj[2])
                    profession = match_obj[3].replace(";", ",").replace("\"", "")

                    # calculate average year
                    avg_year = math.floor((start_year + end_year) / 200) * 100
                    data.append([name, avg_year, profession, url])

                except ValueError:
                    pass

    with open(outfile, 'w', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=';')
        for line in data:
            csv_writer.writerow(line)


def img_downloader(infile, outfile):

    start_time = time()

    data = []
    line_num = 0

    # initialize the driver
    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(options=chrome_options)

    with open(infile, encoding='utf-8') as fh:
        # read all the lines
        lines = fh.readlines()

        for line in lines:
            # print line number
            line_num += 1
            print(line_num)

            # split
            values = line.split(';')

            name = values[0]
            year = int(values[1])
            profession = values[2]
            url = values[3]
            print(url)

            # only filter in specific years: 1500, 1600, 1700
            if year >= 1600 and year < 1700:

                try:
                    # open provided link in a browser window using the driver
                    driver.get(url)

                    # find image filename
                    img_src = driver.find_element_by_css_selector('.image a img').get_attribute("src")
                    img_filename = img_src.split('/')[-2] + ".jpg"

                    # filter out abnormal image filenames
                    if img_filename.startswith('mw'):

                        # find the element
                        img_link = WebDriverWait(driver, 10).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, ".image a"))
                        )

                        # right click
                        ActionChains(driver).context_click(img_link).perform()

                        # choose "save image as..."
                        pyautogui.typewrite(['down', 'down', 'down', 'down', 'down', 'down', 'enter'])
                        sleep(1)

                        # click "save"
                        pyautogui.press('enter')
                        sleep(1)

                        # finally update the data
                        print(name, year, profession, img_filename)
                        data.append([name, year, profession, img_filename])

                finally:
                    with open(outfile, 'w', encoding='utf-8') as csv_fh:
                        csv_writer = csv.writer(csv_fh, delimiter=';')
                        for line in data:
                            csv_writer.writerow(line)

    # close the driver
    driver.close()

    with open(outfile, 'w', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=';')
        for line in data:
            csv_writer.writerow(line)

    end_time = time()
    print("Duration:", (end_time-start_time)/3600, "hours")


if __name__ == "__main__":

    # pre-process items.json
    # input: items.json
    # output: items.csv
    convert_file('items.json', 'items.csv')

    # download all the images
    # img_downloader('items.csv', 'images_1600.csv')