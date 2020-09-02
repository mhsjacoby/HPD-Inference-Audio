'''
cd "to_maggie"
'''
# csv10
import os
from datetime import datetime
import pandas as pd

from glob import glob
from natsort import natsorted
import numpy as np


# Function creates the "timeframe" for 1 day, in 1 sec(changable) freq. 
def create_timeframe(file_path):
	# "filepath" example: G:\H1-black\BS5\csv\H1_BS5_2019-02-23.csv
	file_name = os.path.basename(file_path)  # csv file name with extension
	index = file_name.find("-")
	start_date = file_name[index-4:index+6]  # takes the "2019-02-23" part
	# other way (replace last 2 lines): file_name.split('_')[-1].strip('.csv')
	end_date = datetime.strptime(start_date, '%Y-%m-%d')
	start_date = datetime.strptime(start_date, '%Y-%m-%d')
	end_date = end_date + pd.Timedelta(days=1)
# 	time_frame = pd.date_range(start_date, end_date, freq = '10S').strftime('%Y-%m-%d %H:%M:%S').tolist()
	time_frame = pd.date_range(start_date, end_date, freq = '10S').strftime('%Y-%m-%d %H:%M:%S')
	time_frame = time_frame[:-1]
	return time_frame, file_name



def create_timeframe(start_date, end_date=None, freq="10S"):
	'''
	v2 takes in string of date with %Y-%m-%d format
	Updated with args: end_date and freq.
	
	v1 takes in full csv path:
	"filepath" example: G:\H1-black\BS5\csv\H1_BS5_2019-02-23.csv
	'''
	start_date = datetime.strptime(start_date, '%Y-%m-%d')
	if end_date == None:	
		# end_date = datetime.strptime(start_date, '%Y-%m-%d')
		end_date = start_date + pd.Timedelta(days=1)
	else:
		end_date = datetime.strptime(end_date, '%Y-%m-%d')
		
	time_frame = pd.date_range(start_date, end_date, freq=freq).strftime('%Y-%m-%d %H:%M:%S')
	time_frame = time_frame[:-1]

	return time_frame	










# ==== Add Arg parsers ====

H_num = 6
station_color = "B"
station_nums = [2,3,4]
# station_nums = [2]


if station_color == "B":
	sta_col = "black"
elif station_color == "R":
	sta_col = "red"
elif station_color == "G":
	sta_col = "green"


# for station_num in station_nums:
station_num = station_nums[0]

data_path = f"C:/Users/Sin Yong Tan/Desktop/to_maggie/H{H_num}-{sta_col}/Inference_DB/{station_color}S{station_num}/audio_inf"
save_path = os.path.join(data_path,"processed")

# Create Folder
if not os.path.exists(save_path):
	os.makedirs(save_path)


days = natsorted(glob(os.path.join(data_path,"*.csv")))

# for day in days:
day = days[0]
 	# print(day)

	
# Read data
data = pd.read_csv(day,index_col=0)
data.index = pd.to_datetime(data.index)
data = data[~data.index.duplicated(keep='first') # remove duplicates

# Complete data
timeframe, fname = create_timeframe(day)
timeframe = pd.to_datetime(timeframe)


data = data.reindex(timeframe, fill_value=np.nan)

# Shift the data by 1
data = data.shift(periods=1)

# Save it
data.index.name = "timestamp"
data.to_csv(os.path.join(save_path,fname))




