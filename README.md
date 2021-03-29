datasets/english -> Contains all datasets used in this project, split in dev test and train

datasets/simplified -> Contains the simplified version of the dev dataset. This has been done as an experiment early in the start of the project to see how the data 
used in the complex word identification phase would look like simplified

frequency/word-freq-eng.pkl -> Contains the frequency of the English words

frequency/freq-preprocess.pkl -> Just a preprocessing of the frequencies in order to be used in the baseline model.

pretrained_models -> Pretrained models of Google News vectors trained on word2vec
You need to download them from here:<br/>
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit<br/>
http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz

utils/baseline -> Contains all the models and, the feature extraction process

utils/dataset -> Reads data from dataset

utils/datasimplifier -> Simplfies the senteces from the dataset

utils/phrasesimplifier -> Given a phrase, returns the same phrase but simplified

utils/scorer -> Computes the f1 score

utils/simplfier -> Contains the simplifier model which is based on BERT Masked LM

utils/textsimplifier -> Simplifies a given text

utils/wordvec -> Contains all the models that are trained based on word2vec output

feature_based_solution -> Contains the solution that relies on the feature extraction process. 
It can perform complex word identification as well as lexical simplification

word2vec_solution -> Contains the solution that relies on word2vec. It can do only complex word identification.
utils/hard_vote_all_models -> Hard votes with all models (Complex word identification)

-----------------------------------------------------------------------

In order to run the project:

Import the environment.yml into your environment. I used anaconda. You can simply replicate my 
environment by executing "conda env create -f environment.yml". 

For word2vec solution you need  to download the models specified above.

You can try your examples by opening jupyter lab and replace the text that you want to simplify

In order to assess the complex word identification part, you need to run hard_vote_all_models


