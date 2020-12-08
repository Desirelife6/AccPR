# TransASTNN

Transformed ASTNN for similiar code detection

### Requirements

+ python 3.6<br>
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 1.0.0<br> (The version used in our paper is 0.3.1 and source code can be cloned by specifying the v1.0.0 tag if needed)
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install

Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0

Install pytorch according to your environment, see https://pytorch.org/ 

### Project Structure

#### base_model & base_train

Model with segmentation & training process

#### base_embedding

Word2Vec obtained by original astnn

#### all_words_embedding

Word2Vec obtained by our own segmentation method & our corpus

#### base_result

Models and logs of **base_train.py**

#### Predict.py

Used for predicting similarities between code snippets