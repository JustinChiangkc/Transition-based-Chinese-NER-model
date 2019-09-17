Environment:
1. python3.6
2. Pytorch1.0.0


<Run the API>
-File name:
	run.py

-Instructions:
	
	1. Parameter setting:
		a. --model_path "../save_model/save_XX/model.pth"
			i.   save_65: weiboNER
			ii.  save_68: ontonotesNER
			iii. save_75: insurance_conversationNER
	2. Run "run.py" and send POST requests to your [URL]/ner_tagger to check NER result, ex: (http://doraemon.iis.sinica.edu.tw/ner_tagger/)


<Test model performance on corpus to see F1 score>
-File name:
	test.py

-Instructions:
	
	1.parameters setting:
		a. model selection
			--model_path "../save_model/save_XX/model.pth"
			i.   save_65: weiboNER
			ii.  save_68: ontonotesNER
			iii. save_75: insurance_conversationNER
		b. test set selection
			--testset_path "../data/insurance_dataset/test"
			i.   "../data/weiboNER/test" : weiboNER testset
			ii.  "../data/ontonotes-release-4.0/test" : ontonoteNER testset##################################
			iii. "../data/insurance_dataset/test" : insurance conversation NER testset

	2. Run "test.py" and check F1 score.


<Train new model>
-File name:
	main.py

-Instructions:
	
	1.parameters setting:
		a. data selection:
			--data_path "../data/insurance_dataset/"
			i.   "../data/weiboNER/" : weiboNER dataset
			ii.  "../data/ontonotes-release-4.0/" : ontonoteNER dataset
			iii. "../data/insurance_dataset/" : insurance conversation NER dataset

	2. Run "main.py" to train.



<Generate Configuration>
-File name:
	Oracle_NER.py

-Instructions:
	
	1.parameters setting:
		a. Original NER tagged corpus:
		 	-f NER_tagged_corpus 
		 	**original NER tagging dataset in CoNLL NER tagging format and BIO scheme
		 		ex: 
			 		你	O
					從	O
					日	B-GPE
					本	I-GPE
					回	O
					來	O
					嘛	O 
		b. File for configuration:
			-o file_for_configuration
			**Configuration would be from 0.json to n.json (each .json represent a sentence)

