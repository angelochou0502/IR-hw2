import numpy as np
import sys
import plsa_2

MIN_COUNT = 200
MAX_ITER = 100
NUMBER_OF_TOPIC = 40
DICTIONARY_PATH = './data/good_dictionary.csv'

def main():
	doc_path = sys.argv[1]
	group_path = sys.argv[2]
	out_path = sys.argv[3]

	document = plsa_2.Document(doc_path)
	document.get_doc_dict()
	document.get_vocabulary(MIN_COUNT)
	document.plsa(NUMBER_OF_TOPIC , MAX_ITER)

	group = plsa_2.Group(group_path)
	if(sys.argv[4] == '1'):
		group.add_dictionary(DICTIONARY_PATH)
	group.predict( document.vocabulary , out_path)

	return


if __name__ == "__main__":
	main()