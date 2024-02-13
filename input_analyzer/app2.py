from flask import Flask, render_template, request, jsonify
import botUtils
from botUtils import message_analysis
from botUtils import load_database

import re

app = Flask(__name__)
USER_INPUT=""
RESPONSE_MESSAGE=""
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/send_message", methods=['POST'])
def send_message():
    USER_INPUT = request.form["message"]
    RESPONSE_MESSAGE =  do_somenthing(USER_INPUT)
    
    return jsonify({"message": RESPONSE_MESSAGE})

def do_somenthing(USER_INPUT):
    return message_analysis(USER_INPUT)
    
                                                   
if __name__ == "__main__":
    load_database()
    app.run(debug=True)

    
