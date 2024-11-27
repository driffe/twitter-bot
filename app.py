from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import get_client
from user import User
from dataset import analyze_user_data
from bot_detection import detect_bot
import asyncio

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request:", request.json) 
    user_input = request.get_json()
    screen_name = user_input.get('screen_name')
    print("Screen name:", screen_name)  
    result = asyncio.run(process_prediction(screen_name))
    print("Result:", result)  
    return jsonify(result)

async def process_prediction(screen_name):
    try:
        client = get_client()
        user = await client.get_user_by_screen_name(screen_name)
        print(f"Retrieved user: {user}")

        if not user:
            return {"error": "Invalid User"}

        user_data = await analyze_user_data(user, "UNKNOWN")
        print(f"Analyzed user data: {user_data}")

        if not user_data:
            return {"error": "Could not analyze user data"}

        bot_status = detect_bot(user_data)
        print(f"Bot status: {bot_status}")

        return {"bot_status": bot_status}
    
    except Exception as e:
        print(f"Exception caught: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)