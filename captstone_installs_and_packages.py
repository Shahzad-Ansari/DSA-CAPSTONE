# -*- coding: utf-8 -*-


!pip install geopandas
!pip install meteostat

from google.colab import files
import pandas as pd
import time
from geopy.geocoders import Nominatim
import numpy as np
from tabulate import tabulate
from google.colab import drive
import geopandas as gpd
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily,Hourly

drive.mount('/content/drive', force_remount=True)

