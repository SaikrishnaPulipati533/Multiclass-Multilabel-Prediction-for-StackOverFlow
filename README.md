
# Multiclass Multilabel Prediction For StackOverflow Questions

**Data set** : https://www.kaggle.com/saikrishnapulipati/multilable-multiclass-prediction-for-stackoverflow/data


**Objective** : predict tags associated with them.

This is a scaled down version of predecting only top 10 most occurring tags 

**Programming Language** : Python using nltk & Keras

**Model Architecture** : Deep Learning using Recurrent Neural Network (RNN)

**About Data Set** 

Dataset has text of questionsand thier corresponding tags from the Stack Overflow programming Q&A website.

This is organized as three files:

1. Questions contains the title, body, creation date, closed date (if applicable), score, and owner ID for all non-deleted Stack Overflow questions.

2. Tags contains the tags on each of these questions.

3. Answers contains the body, creation date, score, and owner ID for each of the answers to these questions. The ParentId column links back to the Questions table. *We don't use this file as we want to predict Tags given a question*
**Data Pre-Processing**

>Tags File
*Code* : Stackoverflow Tags Map & Model.ipynb

1. Read Tags File
2. Identify top 10 Tags by count
3. Manipulate the tags dataframe so that all the Tags for an ID are as a list in a row (grouped by Question ID).
Our Dataset would now have only Id, Title, Body & Tags

>Text Preprocessing
*Code* : Stackoverflow Tags Map & Model.ipynb

We will use nltk, preprocessing from Keras and sklearn to process the text data

*Tags preprocesing*
Use MultiLabelBinarizer from sklearn on the Class labels(Tags)
```python
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(total.Tags)
print(multilabel_binarizer.classes_)

array(['android', 'c#', 'c++', 'html', 'ios', 'java', 'javascript','jquery', 'php', 'python'], dtype=object)
```
*Title & Body Preprocessing*
1. Tokenize the words
2. Convert the tokenized words to sequences

**Model Building** 

Implemented a Hybrid model in TensorFlow using Keras as high level api. Architecture used is RNN. In this model first we train a model using the Title data, then train a model using the Body data. Outputs of both are concatenated and passed thorugh the dense layers before connecting to the output layer

*RNN Model* : The model first uses GRU for the sequence data training with 2 GRU layers one for Title and other for Body. 

RNN for Title has
  - 1 Embedding Layer has input of Title vocabulary length(68969) + 1(for 0 padding) and out put of 2000 embeddings (for better results use full vocabulary length+1)
  - 1 Gated recurrent unit (GRU) layer
  - 1 dense output layer of shape 10(No of classes(tags) we are trying to predict) 
RNN for Body has
  - 1 Embedding Layer has input of Title vocabulary length(1292018) + 1(for 0 padding) and out put of 170 embeddings (for better results use full vocabulary length+1)
  - 1 Gated recurrent unit (GRU) layer
Combine the 2 GRU outputs

The fully connected network has
  - 2 Dense Layers 
  - 1 Dropout layer
  - 1 BatchNormalization layer
  - 1 Dense Output layer
  *Model Compilattion with optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')*

**Model Performance Review**

*Classification Report to check Precision, Recall and F1 Score*

The Model seem to performing good enough with score of 84%. Increase in the Embedding, GRU and dense layers would help in getting better results
**Save the Model & Weights**

Saving the model for transfer learning or model execution later

```python
model.save('./stackoverflow_tags.h5')
```



