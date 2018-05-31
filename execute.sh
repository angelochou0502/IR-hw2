#!/bin/bash
use_dic=0
curl -L -o data/doc.csv https://www.dropbox.com/s/qdntpy7d0p20khd/doc.csv?dl=1
curl -L -o data/group.csv https://www.dropbox.com/s/2dlc8j0qzfwltuz/groups.csv?dl=1
while getopts ":ebd:g:o:" opt; do
	case $opt in
		e)
			use_dic=1
			#echo "-e was triggered , $use_dic"
		;;
		b)
			#echo "-b was triggered"
		;;
		d)
			doc_path=$OPTARG
			#echo "-d was triggered , Parameter: $doc_path"
		;;
		g)
			group_path=$OPTARG
			#echo "-g was triggered , Parameter: $group_path"
		;;
		o)
			out_path=$OPTARG
			#echo "-o was triggered , Parameter: $out_path"
		;;
		\?)
			echo "Invalid option: -$OPTARG"
		;;
		:)
			echo "Option -$OPTARG requires an argument."
		;;
	esac
done

python3 main.py $doc_path $group_path $out_path $use_dic
