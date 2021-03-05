import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


app = Flask(__name__)

cat_model = load_model('category_model.h5')
cat_model._make_predict_function()
with open('tokenizer_category.pickle', 'rb') as handle:
    tokenizer_cat = pickle.load(handle)


prod_model = load_model('product_model.h5')
prod_model._make_predict_function()
with open('tokenizer_product.pickle', 'rb') as handle:
    tokenizer_prod = pickle.load(handle)

class_dict = {0:"Active Again",1:"Easy to Bank",2:"Onboarding",3:"Regulatory & Mandatory",4:"Take More"}
prod_dict = {0:"assets",1:"cards",2:"corp mcc",3:"digital banking",4:"investments",5:"liabilities",6:"stocks",7:"yes remit"}

def processing(text):
    sentence = np.array([text])
    sentence1 = tokenizer_cat.texts_to_sequences(sentence)
    sentence1 = pad_sequences(sentence1,padding='post',maxlen=24)
    sentence2 = tokenizer_prod.texts_to_sequences(sentence)
    sentence2 = pad_sequences(sentence2,padding='post',maxlen=24)
    category = cat_model.predict_classes(sentence1)
    cat_output = class_dict[category[0]]
    product = prod_model.predict_classes(sentence2)
    prod_output = prod_dict[product[0]]
    return cat_output,prod_output





# sentence = np.array(["Special off on lifestyle last Debit/Credit"])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form.get("email_subject")

    # split the text to get each line in a list
    text2 = text.split('\n')

    text_changed = ''.join([line for line in text2])
    #temp = request.form['email_subject']
    category, product = processing(text_changed)
    
    return render_template('index.html', prediction_text='Category sould be  {} and Product should be {}'.format(category,product))


if __name__ == "__main__":
    app.run(debug=True)

