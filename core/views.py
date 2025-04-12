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
import hashlib
import time

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
    variants = request.data.get('variants', 'single')  # 'single' or 'multiple' 
    translation_mode = request.data.get('translation_mode', 'casual')  # 'formal' or 'casual'
    model = request.data.get('model', 'claude').lower()  # 'claude', 'gemini', or 'deepseek'

    if not text_to_translate:
        return Response({"error": "Text to translate is required."}, status=400)

    try:
        if model == 'claude':
            translated_text = translate_with_claude(text_to_translate, source_language, target_language, variants, translation_mode)
        elif model == 'gemini':
            translated_text = translate_with_gemini(text_to_translate, source_language, target_language, variants, translation_mode)
        elif model == 'deepseek':
            translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, variants, translation_mode)
        else:
            return Response({"error": "Invalid model specified"}, status=400)

        return Response({
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_language,
            'target_language': target_language,
            'variants': variants,
            'translation_mode': translation_mode,
            'model': model
        })

    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

def get_cached_translation(text, source_language, target_language, model='claude', variants='single'):
    """
    Check if a translation is already in the cache.
    Returns the cached translation or None if not found.
    """
    # Create a hash of the request parameters
    cache_key = hashlib.md5(f"{text}_{source_language}_{target_language}_{model}_{variants}".encode()).hexdigest()
    
    # Check if we have this in the cache
    cache_ref = db.reference(f'translation_cache/{cache_key}')
    cached_data = cache_ref.get()
    
    if cached_data:
        # If we have a cached result and it's not too old, return it
        timestamp = cached_data.get('timestamp', 0)
        current_time = int(time.time())
        
        # Cache valid for 30 days
        if current_time - timestamp < 30 * 24 * 60 * 60:
            return cached_data.get('translated_text')
    
    return None

def cache_translation(text, source_language, target_language, translated_text, model='claude', variants='single'):
    """
    Store a translation in the cache for future use.
    """
    # Create a hash of the request parameters
    cache_key = hashlib.md5(f"{text}_{source_language}_{target_language}_{model}_{variants}".encode()).hexdigest()
    
    # Store the translation in the cache
    cache_ref = db.reference(f'translation_cache/{cache_key}')
    cache_ref.set({
        'text': text,
        'source_language': source_language,
        'target_language': target_language,
        'translated_text': translated_text,
        'model': model,
        'variants': variants,
        'timestamp': int(time.time())
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_db(request):
    """
    Endpoint for translating text and storing results in Firebase.
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    target_language = request.data.get('target_language', 'en')
    variants = request.data.get('variants', 'multiple')  # Changed from 'mode' to 'variants'
    model = request.data.get('model', 'claude').lower()
    room_id = request.data.get('room_id')
    message_id = request.data.get('message_id')
    is_group = request.data.get('is_group', False)
    translation_mode = request.data.get('translation_mode', 'casual')  # formal or casual

    if not all([text_to_translate, room_id, message_id]):
        return Response({
            "error": "Missing required fields: text, room_id, or message_id"
        }, status=400)

    try:
        # Skip translation if source and target languages are the same
        if source_language == target_language or (source_language == 'auto' and target_language == 'en'):
            # Use original text as translation
            translated_text = text_to_translate.strip('"')
        else:
            # First check if we have this translation cached
            cached_translation = get_cached_translation(
                text_to_translate,
                source_language,
                target_language,
                model,
                variants
            )

            if cached_translation:
                translated_text = cached_translation
            else:
                # If not in cache, get translation from AI model
                if model == 'claude':
                    translated_text = translate_with_claude(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'gemini':
                    translated_text = translate_with_gemini(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'deepseek':
                    translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, variants, translation_mode)
                else:
                    return Response({"error": "Invalid model specified"}, status=400)

                # Cache this translation if it's not a same-language case
                if source_language != target_language:
                    cache_translation(
                        text_to_translate,
                        source_language,
                        target_language,
                        translated_text,
                        model,
                        variants
                    )

        # Process translations and store in Firebase
        translations = process_translations(translated_text, variants)
        
        # Determine the correct Firebase reference path based on is_group flag
        ref_path = 'group_messages' if is_group else 'messages'
        messages_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
        
        if is_group:
            # For group messages, follow the structure with translations field
            update_data = {
                'message': translations['main_translation'],
                'sourceLanguage': source_language,
                'translationMode': translation_mode,
            }
            
            # Add the translation to the translations map using the target language as key
            translations_ref = messages_ref.child('translations')
            translations_ref.update({
                target_language: translations['main_translation']
            })
            
            # Update the main message fields
            messages_ref.update(update_data)
        else:
            # For direct messages, use the original approach
            if variants == 'multiple':
                messages_ref.update({
                    'message': translations['main_translation'],
                    'sourceLanguage': source_language,
                    'translationMode': translation_mode,
                    'messageVar1': translations.get('var1', ''),
                    'messageVar2': translations.get('var2', ''),
                    'messageVar3': translations.get('var3', '')
                })
            else:
                messages_ref.update({
                    'message': translations['main_translation'],
                    'sourceLanguage': source_language,
                    'translationMode': translation_mode,
                })

        return Response({
            'original_text': text_to_translate,
            'translations': translations,
            'source_language': source_language,
            'target_language': target_language,
            'variants': variants,
            'model': model,
            'translation_mode': translation_mode,
            'cached': cached_translation is not None
        })

    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

def process_translations(translated_text, variants='multiple'):
    """
    Process the translated text into variations.
    Returns a dict with main translation and variations.
    """
    # For single variants, just return the main translation
    if variants == 'single':
        cleaned_text = translated_text.strip().strip('"')
        return {
            'main_translation': cleaned_text
        }
        
    # For multiple variants, process variations
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

def translate_with_claude(text, source_language, target_language, variants, translation_mode="casual"):
    """Helper function for Claude translation"""
    # Base translator instruction - can be cached
    base_instruction = {
        "type": "text",
        "text": "You are a direct translator. Your job is to translate text accurately while preserving meaning and tone.",
        "cache_control": {"type": "ephemeral"}
    }
    
    # Apply translation mode (formal vs casual)
    formality_instruction = ""
    if translation_mode == "formal":
        formality_instruction = "Use formal language and respectful tone in your translation. "
    else:
        formality_instruction = "Use casual, conversational language in your translation. "
    
    # Language-specific instructions - will change with each request
    if variants == 'single':
        language_instruction = {
            "type": "text",
            "text": ((f"Translate from {source_language} " if source_language != 'auto' else "") +
                   f"to {target_language}. " +
                   formality_instruction +
                   "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. " +
                   "Preserve any slang or explicit words from the original text.")
        }
    else:
        language_instruction = {
            "type": "text",
            "text": ((f"Translate from {source_language} " if source_language != 'auto' else "") +
                   f"to {target_language} and provide exactly 3 numbered variations. " +
                   formality_instruction +
                   "Output ONLY the translations - no explanations, no language detection notes. " +
                   "Format: 1. [translation]\\n2. [translation]\\n3. [translation]")
        }

    # Create system message as a list of dictionaries
    system_message = [base_instruction, language_instruction]

    message = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=8192,
        temperature=0 if variants == 'single' else 0.7,
        system=system_message,
        messages=[{"role": "user", "content": text}]
    )

    return message.content[0].text.strip() if message.content else "Translation failed."

def translate_with_gemini(text, source_language, target_language, variants, translation_mode="casual"):
    """Helper function for Gemini translation"""
    
    # Apply translation mode (formal vs casual)
    formality_instruction = ""
    if translation_mode == "formal":
        formality_instruction = "Use formal language and respectful tone in your translation. "
    else:
        formality_instruction = "Use casual, conversational language in your translation. "
    
    system_instruction = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + formality_instruction
        + (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            "Preserve any slang or explicit words from the original text."
            if variants == 'single' else
            f"Translate to {target_language} and provide exactly 3 numbered variations. "
            "Output ONLY the translations - no explanations, no language detection notes. "
            "Format: 1. [translation]\\n2. [translation]\\n3. [translation]"
        )
    )

    # Configure generation parameters
    generation_config = types.GenerateContentConfig(
        temperature=0.1 if variants == 'single' else 0.7,
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

def translate_with_deepseek(text, source_language, target_language, variants, translation_mode="casual"):
    """Helper function for DeepSeek translation"""
    
    # Apply translation mode (formal vs casual)
    formality_instruction = ""
    if translation_mode == "formal":
        formality_instruction = "Use formal language and respectful tone in your translation. "
    else:
        formality_instruction = "Use casual, conversational language in your translation. "
    
    system_prompt = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + formality_instruction
        + (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            "Preserve any slang or explicit words from the original text."
            if variants == 'single' else
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

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_batch(request):
    """
    Endpoint for translating text to multiple languages in a single request.
    Results are stored in Firebase for group messages.
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    target_languages = request.data.get('target_languages', [])
    variants = request.data.get('variants', 'single')
    model = request.data.get('model', 'claude').lower()
    group_id = request.data.get('group_id')
    message_id = request.data.get('message_id')
    translation_mode = request.data.get('translation_mode', 'casual')  # formal or casual
    
    if not all([text_to_translate, group_id, message_id]) or not target_languages:
        return Response({
            "error": "Missing required fields: text, group_id, message_id, or target_languages"
        }, status=400)
    
    try:
        # Results dictionary
        translations_results = {}
        
        # Process each target language
        for target_language in target_languages:
            # Skip translation if source and target languages are the same
            if source_language == target_language or (source_language == 'auto' and target_language == 'en'):
                # Use original text as translation
                translated_text = text_to_translate.strip('"')
                translations_results[target_language] = translated_text
                continue
                
            # Check if this translation is in cache
            cached_translation = get_cached_translation(
                text_to_translate,
                source_language,
                target_language,
                model,
                variants
            )
            
            if cached_translation:
                translated_text = cached_translation
            else:
                # If not in cache, get translation from the selected AI model
                if model == 'claude':
                    translated_text = translate_with_claude(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'gemini':
                    translated_text = translate_with_gemini(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'deepseek':
                    translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, variants, translation_mode)
                else:
                    return Response({"error": "Invalid model specified"}, status=400)
                
                # Cache this translation
                cache_translation(
                    text_to_translate,
                    source_language,
                    target_language,
                    translated_text,
                    model,
                    variants
                )
            
            # Process translation result
            processed = process_translations(translated_text, variants)
            translations_results[target_language] = processed['main_translation']
        
        # Get Firebase reference for the group message
        messages_ref = db.reference(f'group_messages/{group_id}/{message_id}')
        
        # Set default translated message (use first translation or original if no translations)
        default_translation = next(iter(translations_results.values())) if translations_results else text_to_translate
        
        # Update the message with translations
        messages_ref.update({
            'message': default_translation,
            'sourceLanguage': source_language,
            'translationMode': translation_mode,
        })
        
        # Update the translations map
        translations_ref = messages_ref.child('translations')
        translations_ref.update(translations_results)
        
        return Response({
            'status': 'success',
            'translations': translations_results,
            'original_text': text_to_translate,
            'source_language': source_language,
            'translation_mode': translation_mode
        })
        
    except Exception as e:
        return Response({"error": f"Batch translation failed: {str(e)}"}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_group(request):
    """
    Endpoint for translating group messages.
    This endpoint handles:
    1. Fetching all languages from group members in Firebase
    2. Translating the message to all needed languages
    3. Saving translations back to Firebase
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    model = request.data.get('model', 'claude').lower()
    group_id = request.data.get('group_id')
    message_id = request.data.get('message_id')
    translation_mode = request.data.get('translation_mode', 'casual')  # formal or casual
    variants = 'single'  # Default to single variant, not multiple
    
    if not all([text_to_translate, group_id, message_id]):
        return Response({
            "error": "Missing required fields: text, group_id, or message_id"
        }, status=400)
    
    try:
        # First, get all members of the group
        group_ref = db.reference(f'groups/{group_id}/members')
        group_members = group_ref.get()
        
        if not group_members:
            return Response({"error": "No members found in group"}, status=404)
        
        # Collect all languages from group members
        target_languages = set()
        for member_id in group_members.keys():
            user_ref = db.reference(f'users/{member_id}/language')
            user_language = user_ref.get()
            
            # Skip if it's the same as source language
            if user_language and user_language != source_language:
                target_languages.add(user_language)
            elif not user_language:
                # Default to English if no language is set
                target_languages.add('en')
        
        # Results dictionary
        translations_results = {}
        
        # Process each target language
        for target_language in target_languages:
            # Skip translation if source and target languages are the same
            if source_language == target_language or (source_language == 'auto' and target_language == 'en'):
                # Use original text as translation
                translated_text = text_to_translate.strip('"')
                translations_results[target_language] = translated_text
                continue
                
            # Check if this translation is in cache
            cached_translation = get_cached_translation(
                text_to_translate,
                source_language,
                target_language,
                model,
                variants
            )
            
            if cached_translation:
                translated_text = cached_translation
            else:
                # If not in cache, get translation from the selected AI model
                if model == 'claude':
                    translated_text = translate_with_claude(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'gemini':
                    translated_text = translate_with_gemini(text_to_translate, source_language, target_language, variants, translation_mode)
                elif model == 'deepseek':
                    translated_text = translate_with_deepseek(text_to_translate, source_language, target_language, variants, translation_mode)
                else:
                    return Response({"error": "Invalid model specified"}, status=400)
                
                # Cache this translation
                cache_translation(
                    text_to_translate,
                    source_language,
                    target_language,
                    translated_text,
                    model,
                    variants
                )
            
            # Process translation result
            processed = process_translations(translated_text, variants)
            translations_results[target_language] = processed['main_translation']
        
        # Also add the original language to translations
        translations_results[source_language] = text_to_translate.strip('"')
        
        # Get Firebase reference for the group message
        messages_ref = db.reference(f'group_messages/{group_id}/{message_id}')
        
        # Set default translated message (use first translation or original if no translations)
        default_translation = next(iter(translations_results.values())) if translations_results else text_to_translate.strip('"')
        
        # Update the message with translations
        messages_ref.update({
            'message': default_translation,
            'sourceLanguage': source_language,
            'translationMode': translation_mode,
        })
        
        # Update the translations map
        translations_ref = messages_ref.child('translations')
        translations_ref.update(translations_results)
        
        return Response({
            'status': 'success',
            'translations_count': len(translations_results),
            'languages': list(translations_results.keys()),
            'original_text': text_to_translate,
            'translation_mode': translation_mode
        })
        
    except Exception as e:
        return Response({"error": f"Group translation failed: {str(e)}"}, status=500)
