# Intelligent Systems Group 19 
## Patient's condition prediction using NLP
**Shubham Shah**<br>
**Dhvani Patel**<br>
**Harsh Raval**<br>
**Anurag Tyagi**<br>
**Keerthi Nallabantu** <br>

### Project Goal
The goal of this project is to analyze the drug reviews using Natural Language Processing. The project will be evaluated using both the accuracy of correct prediction and confusion matrix. There will be predictions for disease like Anxity, Birth Control, Depression, Diabetes, Type 2, etc. The study will be completed in few steps as mentioned below. The first of which we will use the drug review data from drugs.com. This will give an explanation as to how words are extracted can be used to predict the medical condition of a patient. Our group plans to use several types of analyses to help predict the medical condition based on reviews using Bag of words and TFIDF tokenizers and Naive Bayes and Passive Aggressive Classifier.

![image](https://user-images.githubusercontent.com/79810765/164039197-3642934b-374e-4690-a120-f0a35a7314b9.png)

### Research Questions:
- Which words/features are more important to a particular medical condition?
- How different tokenization method affect the reviews dataset and accuracy?
- Which model is more efficient for this particular dataset?
- Which is more accurate stemming vs lemmitization?

### Open Notebook in Colab
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamshah207/Disease-Prediction-using-reviews-IS/blob/master/main.ipynb)

**This notebook will contain below action items**
- Tokenize the reviews.
- Clean them by removing puntuation, special characters, numbers.
- Convert them to lower case for better accuracy.
- Lemmitization in place of stemming.
- Two different methods to verctorize: Bag of words & TFIDF.
- Apply two most accurate for text dataset machine learning algorithms: Naive Bayes & Passive Agressive Classifier.
- Compare both models and choose the most accurate one. 

![image](https://user-images.githubusercontent.com/79810765/164042556-296463f0-6708-4497-a78c-a97c9e2065fc.png)


![image](https://user-images.githubusercontent.com/79810765/164039567-ab44cc2f-be50-4f9b-bc08-a9621d6b77cf.png)

## Dataset 
Source Reference: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
- The dataset represent the reviews of the drugs extracted from the site Drugs.com
- There are total six features in the given dataset, which includes 215063 number of records.

- Unnamed:0 Extra column, can be removed as not needed.
- ![image](https://user-images.githubusercontent.com/79810765/164042712-d8c7c4ea-770a-428e-92c8-42c64b2b525d.png)

## Data Preprocessing and Data Cleaning
- As there are no missing values, we do not need to deal with that.
- As there are many classifications but the one which have more reviews we will consider. So we will consider only below conditions to be predicted.
- Reducing the data by limiting the conditions as there are less data for other conditions and due to which model can be skewed.
- As we are using NLP to predict, we will be only considering the reviews and condition columns required to predict.
- As these reviews are extracted from a website, there can be many unnecessary characters, web tags, spaces, letters, special characters, which we need to remove to make the dataset clean.
- Removing unnecessary quotes from all the columns.
- Removing stop words as they do not add any meaning to the reviews.
- Lemmatization is more accurate as there is a meaning and in stemming sometimes there won't be any meaning hence we will be using Lemmatization in place of stemming.
  https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
  https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
- To remove html tags from review, we can use BeautifulSoup module.
- There won't be any meaning for the single letters in the review so we can remove them.
- Removing the stopwords from reviews and making it lemmatized words to that only meaningful words should be present in the reviews.

## Exploratory Data Analysis
### Word Cloud for Birth Control
![image](https://user-images.githubusercontent.com/79810765/164041432-2007ed45-bee1-40f2-919c-fce45e5bc366.png)

### Word Cloud for Depression
![image](https://user-images.githubusercontent.com/79810765/164041516-8f6a242c-5f7b-4ded-b21a-d673dc55f7e6.png)

### Word Cloud for Pain
![image](https://user-images.githubusercontent.com/79810765/164041564-f0fda02b-b95e-48ff-b96d-b56d4012ff0a.png)

### Word Cloud for Anxiety
![image](https://user-images.githubusercontent.com/79810765/164041633-55450a31-0852-4f35-94ac-30f3b043349a.png)

### Word Cloud for Acne
![image](https://user-images.githubusercontent.com/79810765/164041713-48d4e8ec-75b3-41bb-b53c-5e0b2c74107f.png)

### Word Cloud for Bipolar Disorder
![image](https://user-images.githubusercontent.com/79810765/164041759-b46dfed3-83a8-4884-b3e6-6b084603e44a.png)

### Word Cloud for Insomania
![image](https://user-images.githubusercontent.com/79810765/164041802-2b999ab7-1b9a-4a4a-878e-5a953beb4ddd.png)

### Word Cloud for Weight Loss
![image](https://user-images.githubusercontent.com/79810765/164041846-45bfbee2-73f4-46f8-b07d-0bdfb8444713.png)

### Word Cloud for Obesity
![image](https://user-images.githubusercontent.com/79810765/164041892-88e40d00-02f2-478b-8d10-f8bc2c0ed0a7.png)

### Word Cloud for ADHD
![image](https://user-images.githubusercontent.com/79810765/164041938-c326b82a-7062-40d4-af5e-ad951b3be447.png)

### Word Cloud for Diabetes, Type 2
![image](https://user-images.githubusercontent.com/79810765/164041993-ef44d611-5448-4575-bd12-c6c12d4f7415.png)

### Confusion Matrix for Bag of words + Naive Bayes
![image](https://user-images.githubusercontent.com/79810765/164043989-f8c22a93-f0df-4670-831c-6c44c3473753.png)

### Confusion Matrix for Bag of words + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044265-59eb5591-74fc-4adc-800b-250c56497dd2.png)

### Confusion Matrix for TFIDF + Naive Bayes
![image](https://user-images.githubusercontent.com/79810765/164044335-5a1075b9-a466-4e4a-9371-039f005e6b45.png)

### Confusion Matrix for TFIDF + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044457-6cdbaa4f-6c01-442f-a21a-2d9c7fc0d87c.png)

### Confusion Matrix for TFIDF:Bigrams + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044552-e8d5f8a6-489a-46cc-a289-67242e5c9024.png)

### Confusion Matrix for TFIDF:Trigrams + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044670-0b9eba32-27b3-4bc7-804b-b34e4e5e04ea.png)

### Confusion Matrix for TFIDF:Four Grams + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044870-b59689af-367e-4afc-8139-d3f5d09bbd71.png)

### Confusion Matrix for TFIDF:Five Grams + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044910-4f01fbe0-dedc-4a5c-a346-62d06855efe0.png)

### Confusion Matrix for TFIDF:Six Grams + Passive Agressive
![image](https://user-images.githubusercontent.com/79810765/164044952-3516d354-452d-4cf8-ab1f-07a2c6da03e8.png)

## Models and Accuracy
![image](https://user-images.githubusercontent.com/79810765/164042941-72020d76-a35a-4624-9736-e5619bb3129e.png)

## Data Analysis
- Created confusion matrix for each model mentioned. 
- Implemented the method to find out top 5 important words for a particular class using coefficient of the classifier.
- According to the accuracy, TFIDF with 5 grams and 4 grams have same accuracy. 

## Prediction with actual review of 2020
Review = “This is the best medication for anxiety that I've ever tried. You know what it makes me feel like? Like when I was a child before I developed anxiety. I spoke freely. I didn't worry about being judged. So now with this med, I'm not analyzing every word that I say before I say it while panicking that I might say the \"wrong\" thing. I speak freely and comfortably again. I am myself.”

Model Prediction : "Anxiety”

Actual Condition: “Anxiety” 

Reference: https://www.drugs.com/comments/alprazolam/xanax-for-anxiety.html
 
## Conclusion
The best model is TFIDF with 5 grams using Passive Aggressive Classifier ML model 95 % accurate.

## Future Aspect
The system can recommend appropriate drugs based on the reviews from which classification is already done in this project.
