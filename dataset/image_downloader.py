import os
import shutil
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

def csv_writer(data, outfile, append=False):
    mode = 'a' if append else 'w'
    with open(outfile, mode, encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=';')
        for row in data:
            csv_writer.writerow(row)


def csv_reader(infile):
    with open(infile, mode='r', encoding='utf-8') as csv_fh:
        csv_reader = csv.reader(csv_fh, delimiter=';')
        row_list = []
        for row in csv_reader:
            row_list.append(row)
    return row_list


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

    # write the csv
    csv_writer(data, outfile)


def img_downloader(avg_year, skip_lines, infile, outfile, append):

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
            print("Processing", line_num)

            if line_num <= skip_lines:
                continue

            # split
            values = line.split(';')

            name = values[0]
            year = int(values[1])
            profession = values[2]
            url = values[3]
            print("Processing", url)

            # only filter in specific years: 1500, 1600, 1700
            if year == avg_year:

                try:
                    # open provided link in a browser window using the driver
                    driver.get(url)

                    # find the description
                    img_desp_list = WebDriverWait(driver, 10).until(
                        EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'p.objectDescription'))
                    )
                    delimiter = "\n"
                    img_desp_text = delimiter.join([img_desp.text for img_desp in img_desp_list])

                    # find the image
                    img_src_list = WebDriverWait(driver, 10).until(
                        EC.visibility_of_all_elements_located((By.CSS_SELECTOR, '.eventsItem .image a img'))
                    )
                    # get the image filename
                    img_filename_list = [img_src.get_attribute('src').split('/')[-2] + ".jpg" for img_src in img_src_list]

                    # find the link
                    img_link_list = WebDriverWait(driver, 10).until(
                        EC.visibility_of_all_elements_located((By.CSS_SELECTOR, ".eventsItem .image a"))
                    )

                    # if there are multiple images for one person
                    for img_filename, img_link in zip(img_filename_list, img_link_list):

                        # filter out abnormal image filenames
                        if not img_filename.startswith('mw'):
                            continue

                        # right click
                        ActionChains(driver).context_click(img_link).perform()

                        # choose "save image as..."
                        pyautogui.typewrite(['down', 'down', 'down', 'down', 'down', 'down', 'enter'])
                        sleep(1)

                        # click "save"
                        pyautogui.press('enter')
                        sleep(1)

                    # finally update the data
                    print("Storing", name, year, profession, img_filename_list, img_desp_text)
                    data.append([name, year, profession, img_filename_list, img_desp_text])

                finally:
                    # write the csv
                    csv_writer(data, outfile, append=append)


    # close the driver
    driver.close()

    # write the csv
    csv_writer(data, outfile, append=append)

    end_time = time()
    print("Duration:", (end_time-start_time)/3600, "hours")


def prepare_dataset(avg_year, infile, outfile):

    data = []

    row_list = csv_reader(infile)

    for row in row_list:
        # move images into the directory for each century
        img_filename_list = row[3].strip('][').split(',')

        # create the directory for the century if it does not exist
        dst_dir = os.path.join('images', str(avg_year))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        src_dir = os.path.join('images')

        img_filename_list_refined = []

        # refine the image filename list
        for img_filename in img_filename_list:
            # remove leading and ending spaces, and '
            img_filename = img_filename.strip().strip("'")

            # remove duplicate image files, e.g. 'mw01153 (1).jpg'
            match_obj = re.match(r'(mw\d+)\s*(\(\d+\)).jpg', img_filename, re.M | re.I)

            if match_obj:
                img_filename = str(match_obj[1]) + ".jpg"

            src_img_filename = os.path.join(src_dir, img_filename)
            dst_img_filename = os.path.join(dst_dir, img_filename)

            # move the files
            if os.path.exists(src_img_filename):
                print('Remove from', src_img_filename, 'to', dst_img_filename)
                shutil.copyfile(src_img_filename, dst_img_filename)
                # delete the source files
                os.remove(src_img_filename)

            # update the files
            img_filename_list_refined.append(img_filename)

        # remove the duplicate files
        img_filename_list_refined = list(set(img_filename_list_refined))

        # extract the valid description
        img_desp_list = row[4].split('\n')
        delimiter = ". "
        description = delimiter.join(img_desp_list)

        data.append([row[0], row[1], row[2], img_filename_list_refined, description])

    # write the csv
    csv_writer(data, outfile)


if __name__ == "__main__":

    data_dir_name = 'data'

    # Step 1
    # pre-process items.json
    # input: items.json
    # output: items.csv
    # convert_file(infile=os.path.join(data_dir_name, 'items.json'), outfile=os.path.join(data_dir_name, 'items.csv'))

    # Step 2
    # download all the images
    avg_year = 1700
    skip_lines = 3053
    append = True
    img_downloader(avg_year=avg_year,
                   skip_lines=skip_lines,
                   infile=os.path.join(data_dir_name, 'items.csv'),
                   outfile=os.path.join(data_dir_name, 'images_' + str(avg_year) + '_raw.csv'),
                   append=append)

    # Step 3
    # prepare the dataset
    # avg_year = 1700
    # prepare_dataset(avg_year=avg_year,
    #                 infile=os.path.join(data_dir_name, 'images_' + str(avg_year) + '_raw.csv'),
    #                 outfile=os.path.join(data_dir_name, 'images_' + str(avg_year) + '.csv'))