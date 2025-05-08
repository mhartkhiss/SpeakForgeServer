# SpeakForge API

This is the backend API for the SpeakForge project built with Django and Django REST Framework.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your environment variables:
```
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url

# API Keys for translation services
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY1=your-gemini-api-key
GEMINI_API_KEY2=your-backup-gemini-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Firebase credentials
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY=your-private-key-with-newlines
FIREBASE_CLIENT_EMAIL=your-client-email
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
FIREBASE_CLIENT_CERT_URL=your-client-cert-url
FIREBASE_DATABASE_URL=your-database-url
```

5. Run migrations:
```bash
python manage.py migrate
```

6. Start the development server:
```bash
python manage.py runserver
```

The API will be available at http://localhost:8000/