import flask
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
with open(f'models/fake_news.pkl','rb') as m:
    model = pickle.load(m)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('main.html'))
@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv('train.csv')
    data1= data[['author','text']].fillna('None')
    cv = CountVectorizer()
    xc = cv.fit_transform(data1['author']+data1['text'])
    if flask.request.method == 'POST':
        author1 = flask.request.form['author']
        author2 = flask.request.form['text']
        dat1 = [author1]
        dat3 = [author2]
        dat2 = pd.DataFrame(list(zip(dat1, dat3)), columns =['author', 'text'])
        vect = cv.transform(dat2['author']+dat2['text']).toarray()
        mypred = model.predict(vect)
    return flask.render_template('result.html',prediction = mypred)


    
if __name__ == '__main__':
    app.run(debug=True)
