import firebase_admin
from firebase_admin import credentials, db
import os
import json
from pathlib import Path

def initialize_firebase():
    try:
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            print("Firebase already initialized")
            return
        
        # Try to use environment variables first
        if os.environ.get('FIREBASE_PROJECT_ID') and os.environ.get('FIREBASE_PRIVATE_KEY'):
            print("Initializing Firebase with environment variables")
            # Create credential dictionary from environment variables
            cred_dict = {
                "type": os.environ.get('FIREBASE_TYPE', 'service_account'),
                "project_id": os.environ.get('FIREBASE_PROJECT_ID'),
                "private_key_id": os.environ.get('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": os.environ.get('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.environ.get('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.environ.get('FIREBASE_CLIENT_ID'),
                "auth_uri": os.environ.get('FIREBASE_AUTH_URI'),
                "token_uri": os.environ.get('FIREBASE_TOKEN_URI'),
                "auth_provider_x509_cert_url": os.environ.get('FIREBASE_AUTH_PROVIDER_CERT_URL'),
                "client_x509_cert_url": os.environ.get('FIREBASE_CLIENT_CERT_URL')
            }
            
            # Initialize Firebase with credentials from environment variables
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.environ.get('FIREBASE_DATABASE_URL')
            })
            print("Firebase initialized successfully with environment variables")
        else:
            # Fall back to JSON file if environment variables are not set
            print("Environment variables not found, trying to use JSON file")
            base_dir = Path(__file__).resolve().parent.parent
            cred_path = base_dir / 'speakforge_firebase.json'
            
            if not cred_path.exists():
                raise FileNotFoundError(f"Firebase credentials file not found at {cred_path} and environment variables are not set")
                
            # Initialize Firebase Admin SDK with JSON file
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.environ.get(
                    'FIREBASE_DATABASE_URL', 
                    'https://appdev-86a96-default-rtdb.asia-southeast1.firebasedatabase.app'
                )
            })
            print("Firebase initialized successfully with JSON file")
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
        raise