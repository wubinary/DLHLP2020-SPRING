#!/bin/bash

download_from_gdrive() {
	file_id=$1
	file_name=$2
			    
	# first stage to get the warning html
	curl -c /tmp/cookies \
	"https://drive.google.com/uc?export=download&id=$file_id" > \
	/tmp/intermezzo.html

	# second stage to extract the download link from html above
	download_link=$(cat /tmp/intermezzo.html | \
	grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
	sed 's/\&amp;/\&/g')
	curl -L -b /tmp/cookies \
	"https://drive.google.com$download_link" > $file_name
}

###############################################
################   my_cws_bert.pt   ###############
#if [ ! -f "OUT_DIR/my_cws_bert.pt" ]; then
echo " [Info] download my_cws_bert.py model "
download_from_gdrive 14TiTTl_LkjWsh2C2JXq9oVxMtZcJnSh4 OUT_DIR/my_cws_bert.pt  
#fi 

