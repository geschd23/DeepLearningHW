import argparse
import random
import json
import itertools
import logging
import re
import os
import uuid
import sys
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup
REQUEST_HEADER = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}


def get_page(url,header):
    return BeautifulSoup(urlopen(Request(url,headers=header)),'html.parser')

def check_args(args):
    if not args.search:
        lines = open("91K nouns.txt").read()
        line = lines[0:]
        words = line.split()
        args.search = random.choice(words)
        print("No search term provided; searching for "+args.search)
    if not os.path.exists(args.des):
        os.makedirs(args.des)


def extract_images_from_soup(soup):
    image_elements = soup.find_all("div", {"class": "rg_meta"})
    metadata_dicts = (json.loads(e.text) for e in image_elements)
    link_type_records = ((d["ou"], d["ity"]) for d in metadata_dicts)
    return link_type_records

def extract_images(query, num_images):
    url = build_query(query)
    soup = get_page(url, REQUEST_HEADER)
    link_type_records = extract_images_from_soup(soup)
    return itertools.islice(link_type_records, num_images)

def save_image(raw_image, image_type, save_directory):
    extension = image_type if image_type else 'jpg'
    file_name = uuid.uuid4().hex
    save_path = os.path.join(save_directory, file_name)
    with open(save_path, 'wb') as image_file:
        image_file.write(raw_image)

def get_raw_image(url):
    req = Request(url, headers= REQUEST_HEADER)
    resp = urlopen(req)
    return resp.read()

def save_image(raw_image, image_type, save_directory):
    extension = image_type if image_type else 'jpg'
    file_name = str(uuid.uuid4().hex) + "." + extension
    save_path = os.path.join(save_directory, file_name)
    with open(save_path, 'wb+') as image_file:
        image_file.write(raw_image)

def download_images_to_dir(images, save_directory, num_images):
    for i, (url, image_type) in enumerate(images):
        try:
            #logger.info("Making request (%d/%d): %s", i, num_images, url)
            raw_image = get_raw_image(url)
            save_image(raw_image, image_type, save_directory)
        except Exception as e:
            #logger.exception(e)
            print(e)

def build_query(query):
    url = "https://www.google.com/search?q=%s&tbm=isch&source=lnt&tbs=isz:ex,iszw:32,iszh:32" % query #Hardcoded 32x32 requirement
    return url

def main():
    dir_path = os.getcwd()#Get where the file is saved
    default_path = os.path.join(dir_path,r'images')
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--search",help="term to search for (String)",type=str)
    parser.add_argument("-n","--num",help="How many images do you want? (Integer)", default=10,type=int)
    parser.add_argument("-d","--des",help="Where do you want to save?(String)",default=default_path,type=str)
    # parser.add_argument("-a","--aspect",help="Specify your height and width of the image",nargs='+',type=int)
    args = parser.parse_args()
    query = args.search
    num_images = args.num
    save_dir = args.des
    check_args(args)
    images = extract_images(query,num_images)
    download_images_to_dir(images, save_dir, num_images)

if __name__ == '__main__':
    main()
