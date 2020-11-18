"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
img = Image.open("explore.jpg")
st.image(img,width =600)
# Data dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import re
from nltk.probability import FreqDist
import itertools
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

def word_cloud(df):
	pro_words = ' '.join([text for text in df['message'][df['sentiment']==1]])
	pro_wordcloud = WordCloud(width=400,height=250,random_state=73,max_font_size=110,background_color='white',colormap="Greens").generate(pro_words)

	neutral_words = ' '.join([text for text in df['message'][df['sentiment']==0]])
	neutral_wordcloud = WordCloud(random_state=73,  max_font_size=110,  background_color='white', colormap="Purples").generate(neutral_words)

	anti_words = ' '.join([text for text in df['message'][df['sentiment']==0]])
	anti_wordcloud = WordCloud(random_state=73,max_font_size=110, background_color='white',colormap="Reds").generate(anti_words)

	news_words = ' '.join([text for text in df['message'][df['sentiment']==0]])
	news_wordcloud = WordCloud(random_state=73, max_font_size=110, background_color='white',colormap="Blues").generate(news_words)


	fig, ax = plt.subplots(2,2, figsize=(15,10))
	ax[0,0].imshow(pro_wordcloud, interpolation="bilinear")
	ax[0,1].imshow(anti_wordcloud, interpolation="bilinear")
	ax[1,0].imshow(neutral_wordcloud, interpolation="bilinear")
	ax[1,1].imshow(news_wordcloud, interpolation="bilinear")

	# Remove the ticks on the x and y axes
	for axs in fig.axes:
		plt.sca(axs)
		plt.axis('off')

	ax[0,0].set_title('Pro climate change\n', fontsize=20)
	ax[0,1].set_title('Anti climate change\n', fontsize=20)
	ax[1,0].set_title('Neutral\n', fontsize=20)
	ax[1,1].set_title('News\n', fontsize=20)
	#plt.tight_layout()
	plt.show()
	st.pyplot(fig)

def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages



	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information","Exploratory Data Analysis","Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("A group of data science students building a classification model to classify sentiments on twitter data as either negative, positive, neatral or news.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	#building "Exploratory Data Analysis" page

	if selection == "Exploratory Data Analysis":
		st.info("Exploratory Data Analysis With Visualizations")
		#word_cloud(raw)
		length = [len(text) for text in list(raw['message'].astype('str'))]
		# Plot the distribution of the length tweets for each class using a box plot
		sns.boxplot(x=raw['sentiment'], y=length, data=raw, palette=("Blues_d"))
		plt.title('Distribution of wordcounts for each class')
		plt.ylim(0,240 )
		plt.show()
		st.pyplot()
		
		word_cloud(raw)

		# Display target distribution
		fig, axes = plt.subplots(ncols=2,nrows=1,figsize=(15, 5), dpi=100)
		sns.countplot(raw['sentiment'], ax=axes[0])
		labels=['Pro (1)', 'News (2)', 'Neutral (0)', 'Anti (-1)'] 
		axes[1].pie(raw['sentiment'].value_counts(),labels=labels,autopct='%1.0f%%',shadow=True,startangle=90,explode = (0.1, 0.1, 0.1, 0.1))
		fig.suptitle('Tweet class distribution', fontsize=20)
		plt.show()
		st.pyplot(fig)


	# Building out the predication page
	if selection == "Prediction":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")
		st.info("Prediction with ML Models")
		multiple_files = st.file_uploader('Label', accept_multiple_files=True)
		for file in multiple_files:
			file_body = file.read()
			file.seek(0)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		models = ["Logistic Regression","Naive Bayes","Linear SVC","Random Forest","AdaBoost Classifier","Decision Tree"]
		model_selector = st.selectbox("Choose classification Model",models)
		prediction_labels = {#Prediction label dictionary for output
		'Anti':-1,
		'Neutral':0,
		'Pro':1,
		'News':2
		
		}
	
		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/LR_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])

			#predictor = joblib.load(open(os.path.join("resources/NB_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])

			predictor = joblib.load(open(os.path.join("resources/LSVC_model.pkl"),"rb"))
			prediction = predictor.predict([tweet_text])

			#predictor = joblib.load(open(os.path.join("resources/AC_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])

	#elif model_selector=="Decision Tree":

			#predictor = joblib.load(open(os.path.join("resources/DT_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])

	#elif model_selector=="Random Forest":

			#predictor = joblib.load(open(os.path.join("resources/RF_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])
			#predictor = joblib.load(open(os.path.join("resources/DT_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])
			#predictor = joblib.load(open(os.path.join("resources/RF_model.pkl"),"rb"))
			#prediction = predictor.predict([tweet_text])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			final_result = show_pred_label(prediction,prediction_labels)
			st.success("Text Categorized as: {}".format(prediction))	

	






# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
