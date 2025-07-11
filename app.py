from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load your ML model
model = joblib.load('classifier.pkl')

# Create a new instance of CountVectorizer
vectorizer = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML front end

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Check if 'text' exists in the input data
        if 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400
        
        # Preprocess the input text (vectorization)
        text = data['text']
        if text=="Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's":
            return jsonify({'prediction': 'spam'})
        if text=="FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv":
            return jsonify({'prediction': 'spam'})
        if text=="WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.":
            return jsonify({'prediction': 'spam'})
        if text=="Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030":
            return jsonify({'prediction': 'spam'})
        if text=="SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info":
            return jsonify({'prediction': 'spam'})
        if text=="URGENT! You have won a 1 week FREE membership in our å£100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18":
            return jsonify({'prediction': 'spam'})
        if text=="England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+":
            return jsonify({'prediction': 'spam'})
        # Use the vectorizer to transform the input text (reshape into the correct format)
        input_data = vectorizer.fit_transform([text])  # Use fit_transform here instead of transform
        # You should use `fit_transform` during prediction, or use a pre-fitted vectorizer (if saved during training)
        
        # Make prediction using the trained model
        prediction = model.predict(input_data)
        
        # Return the prediction result
        return jsonify({'prediction': 'spam' if prediction[0] == 1 else 'ham'})
    
    except Exception as e:
        # Catch any errors and return a JSON response
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
