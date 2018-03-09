from bs4 import BeautifulSoup
import urllib.requests as urllib2
import requests
import re
import os
import argparse
import sys
import json

def get_page(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,heaaders=header)),'html.parser')

# def check_args(args):


def main(args):
    dir_path = os.getcwd()#Get where the file is saved
    default_path = os.path.join(dir_path,r'images')
    if not os.path.exists(default_path):
        os.makedirs(default_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--search",help="term to search for (String)",type=str)
    parser.add_argument("-n","--num",help="How many images do you want? (Integer)", default=10,type=int)
    parser.add_argument("-d","--des",help="Where do you want to save?(String)",default=default_path,type=str)
    args = parser.parse_args()
    
