import firebase_admin
from firebase_admin import credentials, db
import os
from pathlib import Path

def initialize_firebase():
    # Get the path to the service account file
    base_dir = Path(__file__).resolve().parent.parent
    cred_path = base_dir / 'speakforge_firebase.json'
    
    try:
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://appdev-86a96-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}") 