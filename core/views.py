from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.reverse import reverse
import anthropic
import os
import json
import requests
from django.views.generic import TemplateView
from openai import OpenAI
from google import genai
from google.genai import types
from firebase_admin import db
from .models import TranslationCache
from django.db.models import F
from django.core.exceptions import ObjectDoesNotExist

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Initialize DeepSeek client with OpenAI compatibility
deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Initialize Gemini clients with both API keys
gemini_clients = []
for api_key in [os.environ.get("GEMINI_API_KEY1"), os.environ.get("GEMINI_API_KEY2")]:
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            gemini_clients.append(client)
        except Exception as e:
            print(f"Failed to initialize Gemini client with error: {str(e)}")

# Get API keys from environment
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY1"),
    os.environ.get("GEMINI_API_KEY2")
]
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Create your views here.

@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request, format=None):
    """
    API root endpoint that lists all available endpoints.
    """
    return Response({
        'status': 'Welcome to SpeakForge API',
        'version': '1.0.0',
        'endpoints': {
            'translate': reverse('core:translate', request=request, format=format),
        }
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def translate(request):
    """
    Endpoint for translating text using multiple AI models.
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    target_language = request.data.get('target_language', 'en')
    translation_mode = request.data.get('mode', 'single')  # 'single' or 'multiple'
    model = request.data.get('model', 'claude').lower()  # 'claude', 'gemini', or 'deepseek'

    if not text_to_translate:
        return Response({"error": "Text to translate is required."}, status=400)

    try:
        if model == 'claude':
            translated_text = translate_with_claude(text_to_translate, source_language, target_language, translation_mode)
        elif model == 'gemini':
            translated_text = translate_with_gemini(text_to_translate, source_language, target_language, translation_mode)
        elif model == 'deepseek':
            translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, translation_mode)
        else:
            return Response({"error": "Invalid model specified"}, status=400)

        return Response({
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_language,
            'target_language': target_language,
            'mode': translation_mode,
            'model': model
        })

    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

def get_cached_translation(text, source_language, target_language, model, mode):
    """Helper function to get cached translation"""
    try:
        cache = TranslationCache.objects.get(
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            model=model,
            mode=mode
        )
        # Increment the usage counter
        cache.used_count = F('used_count') + 1
        cache.save()
        return cache.translated_text
    except ObjectDoesNotExist:
        return None

def cache_translation(text, source_language, target_language, translated_text, model, mode):
    """Helper function to cache translation"""
    TranslationCache.objects.create(
        source_text=text,
        source_language=source_language,
        target_language=target_language,
        translated_text=translated_text,
        model=model,
        mode=mode
    )

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_db(request):
    """
    Endpoint for translating text and storing results in Firebase.
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    target_language = request.data.get('target_language', 'en')
    translation_mode = request.data.get('mode', 'multiple')
    model = request.data.get('model', 'claude').lower()
    room_id = request.data.get('room_id')
    message_id = request.data.get('message_id')

    if not all([text_to_translate, room_id, message_id]):
        return Response({
            "error": "Missing required fields: text, room_id, or message_id"
        }, status=400)

    try:
        # First check if we have this translation cached
        cached_translation = get_cached_translation(
            text_to_translate,
            source_language,
            target_language,
            model,
            translation_mode
        )

        if cached_translation:
            translated_text = cached_translation
        else:
            # If not in cache, get translation from AI model
            if model == 'claude':
                translated_text = translate_with_claude(text_to_translate, source_language, target_language, translation_mode)
            elif model == 'gemini':
                translated_text = translate_with_gemini(text_to_translate, source_language, target_language, translation_mode)
            elif model == 'deepseek':
                translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, translation_mode)
            else:
                return Response({"error": "Invalid model specified"}, status=400)

            # Cache the new translation
            cache_translation(
                text_to_translate,
                source_language,
                target_language,
                translated_text,
                model,
                translation_mode
            )

        # Process translations and store in Firebase
        translations = process_translations(translated_text, translation_mode)
        
        # Get Firebase reference
        messages_ref = db.reference(f'messages/{room_id}/{message_id}')
        
        if translation_mode == 'multiple':
            messages_ref.update({
                'message': translations['main_translation'],
                'sourceLanguage': source_language,
                'messageVar1': translations.get('var1', ''),
                'messageVar2': translations.get('var2', ''),
                'messageVar3': translations.get('var3', '')
            })
        else:
            messages_ref.update({
                'message': translations['main_translation'],
                'sourceLanguage': source_language
            })

        return Response({
            'original_text': text_to_translate,
            'translations': translations,
            'source_language': source_language,
            'target_language': target_language,
            'mode': translation_mode,
            'model': model,
            'cached': cached_translation is not None
        })

    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

def process_translations(translated_text, mode='multiple'):
    """
    Process the translated text into variations.
    Returns a dict with main translation and variations.
    """
    # For single mode, just return the main translation
    if mode == 'single':
        cleaned_text = translated_text.strip().strip('"')
        return {
            'main_translation': cleaned_text
        }
        
    # For multiple mode, process variations
    # Handle DeepSeek's specific format with literal '\n'
    if '\\n' in translated_text:
        variations = translated_text.split('\\n')
    else:
        variations = translated_text.split('\n')
        
    cleaned_variations = []
    
    # Clean up variations
    for variation in variations:
        # Remove numbering and clean up
        cleaned = variation.strip()
        cleaned = cleaned.lstrip('123456789.) ')
        cleaned = cleaned.strip('"')
        if cleaned:
            cleaned_variations.append(cleaned)
    
    # Ensure we have at least one translation
    if not cleaned_variations:
        return {'main_translation': translated_text.strip()}
    
    # If we have multiple variations, use them
    if len(cleaned_variations) >= 3:
        return {
            'main_translation': cleaned_variations[1],  # Use middle variation as main
            'var1': cleaned_variations[0],
            'var2': cleaned_variations[1],
            'var3': cleaned_variations[2]
        }
    
    # If we only have one translation
    return {
        'main_translation': cleaned_variations[0]
    }

def translate_with_claude(text, source_language, target_language, mode):
    """Helper function for Claude translation"""
    system_message = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            "Preserve any slang or explicit words from the original text."
            if mode == 'single' else
            f"Translate to {target_language} and provide exactly 3 numbered variations. "
            "Output ONLY the translations - no explanations, no language detection notes. "
            "Format: 1. [translation]\\n2. [translation]\\n3. [translation]"
        )
    )

    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0 if mode == 'single' else 0.7,
        system=system_message,
        messages=[{"role": "user", "content": text}]
    )

    return message.content[0].text.strip() if message.content else "Translation failed."

def translate_with_gemini(text, source_language, target_language, mode):
    """Helper function for Gemini translation"""
    system_instruction = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            "Preserve any slang or explicit words from the original text."
            if mode == 'single' else
            f"Translate to {target_language} and provide exactly 3 numbered variations. "
            "Output ONLY the translations - no explanations, no language detection notes. "
            "Format: 1. [translation]\\n2. [translation]\\n3. [translation]"
        )
    )

    # Configure generation parameters
    generation_config = types.GenerateContentConfig(
        temperature=0.1 if mode == 'single' else 0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        system_instruction=system_instruction
    )

    # Try each client until successful
    last_error = None
    for client in gemini_clients:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[text],
                config=generation_config
            )
            
            if response and hasattr(response, 'text'):
                return response.text.strip()
            
        except Exception as e:
            last_error = str(e)
            print(f"Gemini API error with client: {last_error}")
            continue  # Try next client if current one fails
    
    if last_error:
        raise Exception(f"All Gemini API keys failed. Last error: {last_error}")
    else:
        raise Exception("All Gemini API keys failed with unknown error")

def translate_with_deepseek(text, source_language, target_language, mode):
    """Helper function for DeepSeek translation"""
    system_prompt = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            "Preserve any slang or explicit words from the original text."
            if mode == 'single' else
            f"Translate to {target_language} and provide exactly 3 numbered variations. "
            "Output ONLY the translations - no explanations, no language detection notes. "
            "Format: 1. [translation]\\n2. [translation]\\n3. [translation]"
        )
    )

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            stream=False
        )

        content = response.choices[0].message.content.strip()
        # Clean think tags if present
        content = content.replace("<think>", "").replace("</think>", "").strip()
        return content
    except Exception as e:
        raise Exception(f"DeepSeek API error: {str(e)}")

class TranslatorView(TemplateView):
    template_name = 'core/translator.html'
