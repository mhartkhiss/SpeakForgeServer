from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
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
from firebase_admin import db, auth
from .models import TranslationCache, GroupTranslationMemory
from django.db.models import F
from django.core.exceptions import ObjectDoesNotExist
import hashlib
import time
import datetime
import random
import string
from collections import defaultdict
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from rest_framework import status
from rest_framework import permissions

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

# ==============================================
# Translation API Views
# ==============================================

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

def get_translation_instruction(source_language, target_language, variants, translation_mode):
    """
    Helper function to create consistent translation instructions across models
    """
    # Apply translation mode (formal vs casual)
    formality_instruction = ""
    if translation_mode == "formal":
        formality_instruction = "Use formal language appropriate for academic or professional contexts in all variations. When encountering profanity or inappropriate language, rephrase the entire sentence in a professional way - never output placeholders like '[profanity removed]'. Instead, reformulate to convey the same meaning in clean, respectful language. "
    
    # Base instruction for all models
    base_instruction = (
        f"You are a direct translator. "
        + (f"Translate from {source_language} " if source_language != 'auto' else "")
        + f"to {target_language}. "
        + formality_instruction
    )
    
    # Variant specific instruction
    if variants == 'single':
        output_instruction = (
            "Output ONLY the translation itself - no explanations, no language detection notes, no additional text. "
            + ("Preserve any slang or explicit words from the original text." if translation_mode == "casual" else "")
        )
    else:
        output_instruction = (
            f"Translate to {target_language} and provide exactly 3 numbered variations. "
            "Output ONLY the translations - no explanations, no language detection notes. "
            "Format: 1. [translation]\\n2. [translation]\\n3. [translation]"
        )
    
    return base_instruction + output_instruction

def translate_with_claude(text, source_language, target_language, variants, translation_mode="casual"):
    """Helper function for Claude translation"""
    # Base translator instruction
    base_instruction = {
        "type": "text",
        "text": "You are a direct translator. Your job is to translate text accurately while preserving meaning and tone.",
        "cache_control": {"type": "ephemeral"}
    }
    
    # Language-specific instructions
    language_instruction = {
        "type": "text",
        "text": get_translation_instruction(source_language, target_language, variants, translation_mode)
    }

    # Create system message as a list of dictionaries
    system_message = [base_instruction, language_instruction]

    message = anthropic_client.messages.create(
        #model="claude-3-7-sonnet-20250219",
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0 if variants == 'single' else 0.7,
        system=system_message,
        messages=[{"role": "user", "content": text}]
    )

    return message.content[0].text.strip() if message.content else "Translation failed."

def translate_with_gemini(text, source_language, target_language, variants, translation_mode="casual"):
    """Helper function for Gemini translation"""
    
    system_instruction = get_translation_instruction(source_language, target_language, variants, translation_mode)

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
    
    system_prompt = get_translation_instruction(source_language, target_language, variants, translation_mode)

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

def get_translation(text, source_language, target_language, variants, translation_mode, model):
    """
    Centralized function to handle translation with any model
    """
    # Skip translation if source and target languages are the same
    if source_language == target_language or (source_language == 'auto' and target_language == 'en'):
        return text.strip('"')
    
    # Get translation from the selected AI model
    if model == 'claude':
        return translate_with_claude(text, source_language, target_language, variants, translation_mode)
    elif model == 'gemini':
        return translate_with_gemini(text, source_language, target_language, variants, translation_mode)
    elif model == 'deepseek':
        return translate_with_deepseek(text, source_language, target_language, variants, translation_mode)
    else:
        raise ValueError("Invalid model specified")

def process_translations(translated_text, variants='multiple'):
    """
    Process the translated text into variations.
    Returns a dict with main translation and variations.
    """
    # For single variants, return the main translation and set var1 to the same value
    if variants == 'single':
        cleaned_text = translated_text.strip().strip('"')
        return {
            'main_translation': cleaned_text,
            'var1': cleaned_text  # Add var1 to ensure translation1 is created for single variants
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

def update_firebase_message(ref_path, room_id, message_id, translations, source_language, translation_mode, is_group=False, target_language=None):
    """
    Helper function to update Firebase with translation data
    """
    messages_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
    
    if is_group:
        # For group messages, follow the structure with translations field
        # Do not modify translationState for group messages
        update_data = {
            'senderLanguage': source_language,  # Use senderLanguage instead of sourceLanguage
            'translationMode': translation_mode,
        }
        
        # Add the translation to the translations map using the target language as key
        translations_ref = messages_ref.child('translations')
        translations_ref.update({
            target_language: translations['main_translation']
        })
        
        # Update the main message fields (without changing the message content)
        messages_ref.update(update_data)
    else:
        # For direct messages
        if 'var1' in translations and 'var2' in translations and 'var3' in translations:
            # Full regeneration with all three variations
            messages_ref.update({
                'senderLanguage': source_language,
                'translationMode': translation_mode,
                'translationState': 'TRANSLATED'  # Set state to TRANSLATED when translations are added
            })
            
            # Update translations node with all three variations
            translations_ref = messages_ref.child('translations')
            translations_updates = {
                'translation1': translations.get('var1', ''),
                'translation2': translations.get('var2', ''),
                'translation3': translations.get('var3', '')
            }
            translations_ref.update(translations_updates)
        else:
            # Single translation - only update translation1
            messages_ref.update({
                'senderLanguage': source_language,
                'translationMode': translation_mode,
                'translationState': 'TRANSLATED'  # Set state to TRANSLATED when translations are added
            })
            
            # Add single translation to translations node
            translations_ref = messages_ref.child('translations')
            translations_ref.update({
                'translation1': translations['main_translation']
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
        translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)

        return Response({
            'original_text': text_to_translate,
            'translated_text': translated_text,
            'source_language': source_language,
            'target_language': target_language,
            'variants': variants,
            'translation_mode': translation_mode,
            'model': model
        })

    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

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
        # Get translation
        translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)

        # Process translations and store in Firebase
        translations = process_translations(translated_text, variants)
        
        # Determine the correct Firebase reference path based on is_group flag
        ref_path = 'group_messages' if is_group else 'messages'
        
        # Update Firebase
        update_firebase_message(ref_path, room_id, message_id, translations, source_language, translation_mode, is_group, target_language)

        return Response({
            'original_text': text_to_translate,
            'translations': translations,
            'source_language': source_language,
            'target_language': target_language,
            'variants': variants,
            'model': model,
            'translation_mode': translation_mode,
        })

    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": f"Translation failed: {str(e)}"}, status=500)

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
            # Get translation
            translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)
            
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
        
    except ValueError as e:
        return Response({"error": str(e)}, status=400)
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
            # Get translation
            translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)
            
            # Process translation result
            processed = process_translations(translated_text, variants)
            translations_results[target_language] = processed['main_translation']
        
        # Also add the original language to translations
        translations_results[source_language] = text_to_translate.strip('"')
        
        # Get Firebase reference for the group message
        messages_ref = db.reference(f'group_messages/{group_id}/{message_id}')
        
        # Use the original message in the source language for the message field
        original_message = text_to_translate.strip('"')
        
        # Check if this is a new message
        message_data = messages_ref.get()
        
        # Only set the message field for new messages
        if message_data is None:
            # This is a new message, set the message field
            messages_ref.update({
                'message': original_message,
                'senderLanguage': source_language,  # Only use senderLanguage
                'translationMode': translation_mode,
            })
        else:
            # This is an existing message, only update metadata
            messages_ref.update({
                'senderLanguage': source_language,  # Only use senderLanguage
                'translationMode': translation_mode,
            })
        
        # Remove source language from translations since it's redundant with message field
        if source_language in translations_results:
            del translations_results[source_language]
        
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
        
    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": f"Group translation failed: {str(e)}"}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def regenerate_translation(request):
    """
    Endpoint for regenerating a translation while preserving the original translation mode.
    This ensures that formal mode translations remain formal when regenerated.
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    target_language = request.data.get('target_language', 'en')
    variants = request.data.get('variants', 'single')
    model = request.data.get('model', 'claude').lower()
    translation_mode = request.data.get('translation_mode', 'casual')  # formal or casual
    message_id = request.data.get('message_id')
    room_id = request.data.get('room_id')
    is_group = request.data.get('is_group', False)

    if not all([text_to_translate, target_language]):
        return Response({
            "error": "Missing required fields: text or target_language"
        }, status=400)

    try:
        # Set state to TRANSLATING before starting regeneration
        ref_path = 'group_messages' if is_group else 'messages'
        message_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
        message_ref.update({'translationState': 'TRANSLATING'})

        # Get translation
        translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)

        # Process translations
        translations = process_translations(translated_text, variants)
        
        # Update Firebase if message_id and room_id provided
        if message_id and room_id:
            # Determine the correct Firebase reference path based on is_group flag
            ref_path = 'group_messages' if is_group else 'messages'
            
            # Get reference to the message
            message_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
            
            # For group messages, only update the translations map for the target language
            # instead of updating the main message field
            if is_group:
                # Update only the translations map for the target language
                translations_ref = message_ref.child('translations')
                
                # Add each translation variant to the translations map
                if variants == 'multiple':
                    # If there are multiple variants, only update the first one
                    # (since translations map only supports one translation per language)
                    if 'main_translation' in translations:
                        translations_ref.child(target_language).set(translations['main_translation'])
                else:
                    # Single variant
                    if 'main_translation' in translations:
                        translations_ref.child(target_language).set(translations['main_translation'])
                
                # Store variants if provided
                if 'variation1' in translations:
                    # Create a map of translations with the new structure
                    translations_updates = {}
                    translations_updates['translation1'] = translations['variation1']
                    if 'variation2' in translations:
                        translations_updates['translation2'] = translations['variation2']
                    if 'variation3' in translations:
                        translations_updates['translation3'] = translations['variation3']
                    
                    # Update the translations node ONLY
                    translations_ref.update(translations_updates)
                
                # Set state to TRANSLATED after successful update
                message_ref.update({'translationState': 'TRANSLATED'})
            else:
                # For direct messages, use the original update_firebase_message function
                update_firebase_message(ref_path, room_id, message_id, translations, source_language, translation_mode, is_group, target_language)

        return Response({
            'original_text': text_to_translate,
            'translations': translations,
            'source_language': source_language,
            'target_language': target_language,
            'variants': variants,
            'model': model,
            'translation_mode': translation_mode
        })
        
    except ValueError as e:
        # Set state back to null if translation fails
        if message_id and room_id:
            ref_path = 'group_messages' if is_group else 'messages'
            message_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
            message_ref.update({'translationState': None})
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        # Set state back to null if translation fails
        if message_id and room_id:
            ref_path = 'group_messages' if is_group else 'messages'
            message_ref = db.reference(f'{ref_path}/{room_id}/{message_id}')
            message_ref.update({'translationState': None})
        return Response({"error": f"Translation regeneration failed: {str(e)}"}, status=500)

def get_conversation_context(group_id, message_id, max_context_messages=10, source_language=None):
    """
    Get previous messages from the group chat to use as context for translation
    """
    # Get reference to group messages
    messages_ref = db.reference(f'group_messages/{group_id}')
    
    # Get all messages ordered by timestamp
    messages_query = messages_ref.order_by_child('timestamp').get()
    
    if not messages_query:
        return []
    
    # Convert to list and sort by timestamp
    all_messages = []
    for msg_id, msg_data in messages_query.items():
        if 'timestamp' in msg_data and msg_id != message_id:  # Skip current message
            all_messages.append({
                'message_id': msg_id,
                'data': msg_data
            })
    
    # Sort by timestamp
    all_messages.sort(key=lambda x: x['data'].get('timestamp', 0))
    
    # Get only the last N messages before the current one
    context_messages = all_messages[-max_context_messages:] if len(all_messages) > max_context_messages else all_messages
    
    # Create a mapping of user IDs to anonymous numbered names
    user_id_map = {}
    current_user_number = 1
    
    # Format context for AI models
    formatted_context = []
    for msg in context_messages:
        msg_data = msg['data']
        
        # Get message in the source language from translations map
        message_text = msg_data.get('message', '')
        
        # If source language is specified and we have translations, use the specific language
        if source_language and 'translations' in msg_data and source_language in msg_data['translations']:
            message_text = msg_data['translations'][source_language]
        # If we don't have the translation in source language but have sender language
        elif 'translations' in msg_data and 'senderLanguage' in msg_data:
            sender_language = msg_data.get('senderLanguage')
            if sender_language and sender_language in msg_data['translations']:
                message_text = msg_data['translations'][sender_language]
        
        # Get the sender ID and assign an anonymous user name
        sender_id = msg_data.get('senderId', 'unknown')
        
        # If this user hasn't been seen before, assign a new number
        if sender_id not in user_id_map:
            user_id_map[sender_id] = f"User {current_user_number}"
            current_user_number += 1
            
        # Use the anonymous user name in the context
        anonymous_name = user_id_map[sender_id]
        formatted_context.append(f"{anonymous_name}: {message_text}")
    
    return formatted_context

def translate_with_context(text, source_language, target_language, context, model, translation_mode="casual"):
    """Helper function for translation with conversation context"""
    
    # Make sure we have text to translate
    if not text:
        return "Translation failed - empty text"
        
    # Clean up text if it has quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    # Construct the system instruction to include context
    context_text = ""
    if context and len(context) > 0:
        context_text = "\n".join(context)
        print(f"Using context with {len(context)} messages")
        print(f"Context messages:\n{context_text}")
    else:
        print("No context provided for translation")
    
    # Create base translation instruction
    base_instruction = get_translation_instruction(source_language, target_language, 'single', translation_mode)
    
    # Add context instruction
    context_instruction = base_instruction
    if context_text:
        context_instruction = (
            f"Before translating, consider the following conversation context:\n\n"
            f"{context_text}\n\n"
            f"Now translate the following text, maintaining context and conversational flow "
            f"from {source_language} to {target_language}. "
            f"Only output the translation, no explanations or additional text."
        )
    
    print(f"\n===== CONTEXT-AWARE TRANSLATION =====")
    print(f"Source language: {source_language}")
    print(f"Target language: {target_language}")
    print(f"Text to translate: {text}")
    print(f"Translation mode: {translation_mode}")
    print(f"Model: {model}")
    print(f"Full prompt:\n{context_instruction}\n")
    
    try:
        # Choose which model to use
        if model == 'claude':
            system_message = [
                {
                    "type": "text",
                    "text": base_instruction,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            
            # Add context instruction if there is context
            if context_text:
                system_message.append({
                    "type": "text", 
                    "text": context_instruction
                })
            
            print(f"Claude system message: {system_message}")
            print(f"Claude user message: {text}")
            
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.1,
                system=system_message,
                messages=[{"role": "user", "content": text}]
            )
            
            result = message.content[0].text.strip() if message.content else "Translation failed."
            print(f"Claude response: {result}")
            return result
            
        elif model == 'gemini':
            # Configure generation parameters
            generation_config = types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                system_instruction=context_instruction
            )

            print(f"Gemini system instruction: {context_instruction}")
            print(f"Gemini prompt: {text}")

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
                        result = response.text.strip()
                        print(f"Gemini response: {result}")
                        return result
                    
                except Exception as e:
                    last_error = str(e)
                    print(f"Gemini API error: {last_error}")
                    continue  # Try next client if current one fails
            
            if last_error:
                raise Exception(f"All Gemini API keys failed. Last error: {last_error}")
            else:
                raise Exception("All Gemini API keys failed with unknown error")
                
        elif model == 'deepseek':
            print(f"DeepSeek system instruction: {context_instruction}")
            print(f"DeepSeek prompt: {text}")
            
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": context_instruction},
                        {"role": "user", "content": text}
                    ],
                    stream=False
                )

                content = response.choices[0].message.content.strip()
                # Clean think tags if present
                content = content.replace("<think>", "").replace("</think>", "").strip()
                print(f"DeepSeek response: {content}")
                return content
            except Exception as e:
                raise Exception(f"DeepSeek API error: {str(e)}")
        else:
            # Fallback to normal translation if model not recognized
            print(f"Unrecognized model '{model}', falling back to regular translation")
            return get_translation(text, source_language, target_language, 'single', translation_mode, 'claude')
    except Exception as e:
        print(f"Context-aware translation error: {str(e)}")
        # Fallback to regular translation if context-aware fails
        try:
            return get_translation(text, source_language, target_language, 'single', translation_mode, 'claude')
        except:
            return f"Translation failed: {str(e)}"

def extract_key_terms(text, source_language):
    """
    Extract key terms/phrases from text for translation memory
    Returns list of terms
    """
    # Simple implementation - can be enhanced with NLP library
    # Remove common punctuation
    text = text.replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ')
    
    # Split words and filter out short words and common words
    words = text.split()
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                   'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                   'from', 'of', 'that', 'this', 'these', 'those', 'it', 'they', 'we', 'he', 'she'}
    
    # Extract single words (length > 3)
    single_terms = [w for w in words if len(w) > 3 and w.lower() not in common_words]
    
    # Extract potential phrases (2-3 adjacent words)
    phrases = []
    for i in range(len(words) - 1):
        if len(words[i]) > 2 and len(words[i+1]) > 2:
            phrases.append(f"{words[i]} {words[i+1]}")
    
    # Combine terms and phrases, limit to avoid overload
    all_terms = single_terms + phrases
    return all_terms[:10]  # Limit to 10 key terms

def update_translation_memory(group_id, text, translated_text, source_language, target_language):
    """
    Update translation memory with key terms from this translation
    """
    # Extract key terms
    key_terms = extract_key_terms(text, source_language)
    
    # Also extract key terms from translated text
    translated_terms = extract_key_terms(translated_text, target_language)
    
    # Only continue if we have same number of terms
    if len(key_terms) == len(translated_terms) and len(key_terms) > 0:
        # Update or create translation memory entries
        for i in range(min(len(key_terms), len(translated_terms))):
            try:
                # Try to get existing memory
                memory, created = GroupTranslationMemory.objects.get_or_create(
                    group_id=group_id,
                    original_term=key_terms[i],
                    source_language=source_language,
                    target_language=target_language,
                    defaults={'translated_term': translated_terms[i]}
                )
                
                # If it already existed, update usage count
                if not created:
                    memory.usage_count += 1
                    memory.translated_term = translated_terms[i]
                    memory.save()
                    
            except Exception as e:
                # Log error but continue with other terms
                print(f"Error updating translation memory: {str(e)}")
                continue

def translate_with_batch_context(text, source_language, target_languages, context, model, translation_mode="casual"):
    """Helper function for translation with conversation context for multiple languages at once"""
    
    # Make sure we have text to translate
    if not text:
        return {"error": "Translation failed - empty text"}
        
    # Clean up text if it has quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    # Construct the system instruction to include context
    context_text = ""
    if context and len(context) > 0:
        context_text = "\n".join(context)
        print(f"Using context with {len(context)} messages")
        print(f"Context messages:\n{context_text}")
    else:
        print("No context provided for translation")
    
    # Create context instruction with multiple languages
    target_languages_list = sorted(list(target_languages))
    languages_formatted = ", ".join(target_languages_list)
    
    print(f"\n===== BATCH CONTEXT-AWARE TRANSLATION =====")
    print(f"Source language: {source_language}")
    print(f"Target languages: {languages_formatted}")
    print(f"Text to translate: {text}")
    print(f"Translation mode: {translation_mode}")
    print(f"Model: {model}")
    
    context_prefix = ""
    if context_text:
        context_prefix = f"""Before translating, consider the following conversation context:

{context_text}

"""
    
    # Build prompt for multiple language translation
    system_instruction = f"""You are a direct translator. Translate the input from {source_language} to multiple languages at once.
{context_prefix}Translate the following text to these languages: {languages_formatted}

Use {translation_mode} language style for all translations.

Format your response exactly as follows, with each language on its own line with the format "Language: translation":
"""

    for lang in target_languages_list:
        system_instruction += f"\n{lang}: [translation in {lang}]"
    
    print(f"Full prompt:\n{system_instruction}\n")
    
    try:
        # Choose which model to use
        if model == 'claude':
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.1,
                system=system_instruction,
                messages=[{"role": "user", "content": text}]
            )
            
            result = message.content[0].text.strip() if message.content else "Translation failed."
            print(f"Claude batch response: {result}")
            
            # Parse the response to get individual translations
            translations = {}
            for line in result.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    language = parts[0].strip()
                    translation = parts[1].strip()
                    translations[language] = translation
            
            # Add source language original text
            translations[source_language] = text
            
            return translations
            
        elif model == 'gemini':
            # Configure generation parameters
            generation_config = types.GenerateContentConfig(
                temperature=0.1,
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
                        result = response.text.strip()
                        print(f"Gemini batch response: {result}")
                        
                        # Parse the response to get individual translations
                        translations = {}
                        for line in result.splitlines():
                            line = line.strip()
                            if not line or ":" not in line:
                                continue
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                language = parts[0].strip()
                                translation = parts[1].strip()
                                translations[language] = translation
                        
                        # Add source language original text
                        translations[source_language] = text
                        
                        return translations
                    
                except Exception as e:
                    last_error = str(e)
                    print(f"Gemini API error: {last_error}")
                    continue  # Try next client if current one fails
            
            if last_error:
                raise Exception(f"All Gemini API keys failed. Last error: {last_error}")
            else:
                raise Exception("All Gemini API keys failed with unknown error")
                
        elif model == 'deepseek':
            try:
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text}
                    ],
                    stream=False
                )

                content = response.choices[0].message.content.strip()
                # Clean think tags if present
                content = content.replace("<think>", "").replace("</think>", "").strip()
                print(f"DeepSeek batch response: {content}")
                
                # Parse the response to get individual translations
                translations = {}
                for line in content.splitlines():
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        language = parts[0].strip()
                        translation = parts[1].strip()
                        translations[language] = translation
                
                # Add source language original text
                translations[source_language] = text
                
                return translations
                
            except Exception as e:
                raise Exception(f"DeepSeek API error: {str(e)}")
        else:
            # Fallback to individual translations if model not recognized
            print(f"Unrecognized model '{model}', falling back to individual translations")
            translations = {}
            for target_language in target_languages:
                translations[target_language] = get_translation(text, source_language, target_language, 'single', translation_mode, 'claude')
            translations[source_language] = text
            return translations
            
    except Exception as e:
        print(f"Batch context-aware translation error: {str(e)}")
        # Fallback to individual translations if batch fails
        try:
            translations = {}
            for target_language in target_languages:
                translations[target_language] = get_translation(text, source_language, target_language, 'single', translation_mode, 'claude')
            translations[source_language] = text
            return translations
        except Exception as fallback_error:
            print(f"Fallback translation also failed: {str(fallback_error)}")
            return {lang: f"Translation failed: {str(e)}" for lang in target_languages}

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_group_context(request):
    """
    Enhanced endpoint for translating group messages with conversation context.
    This endpoint handles:
    1. Fetching previous messages as context
    2. Translating with context awareness to all needed languages in a single API call
    3. Saving translations back to Firebase
    4. Updating translation memory
    """
    text_to_translate = request.data.get('text', '')
    source_language = request.data.get('source_language', 'auto')
    model = request.data.get('model', 'claude').lower()
    group_id = request.data.get('group_id')
    message_id = request.data.get('message_id')
    translation_mode = request.data.get('translation_mode', 'casual')  # formal or casual
    context_depth = request.data.get('context_depth', 5)  # Default to 5 previous messages
    use_context = request.data.get('use_context', True)  # Allow disabling context
    
    print(f"Context translation request: group={group_id}, msg={message_id}, src={source_language}, context_depth={context_depth}")
    
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
            try:
                user_ref = db.reference(f'users/{member_id}/language')
                user_language = user_ref.get()
                
                # Skip if it's the same as source language
                if user_language and user_language != source_language:
                    target_languages.add(user_language)
                elif not user_language:
                    # Default to English if no language is set
                    target_languages.add('en')
            except Exception as e:
                print(f"Error getting language for user {member_id}: {str(e)}")
        
        print(f"Target languages: {target_languages}")
        
        # Get conversation context if enabled
        context = []
        try:
            if use_context:
                context = get_conversation_context(group_id, message_id, context_depth, source_language)
                print(f"Retrieved {len(context)} context messages")
        except Exception as e:
            print(f"Error getting conversation context: {str(e)}")
            # Continue without context if there's an error
        
        # If no target languages or only the source language, skip translation
        if not target_languages or (len(target_languages) == 1 and source_language in target_languages):
            translations_results = {source_language: text_to_translate.strip('"')}
        else:
            # Use the new batch translation function for all languages at once
            translations_results = translate_with_batch_context(
                text_to_translate, 
                source_language, 
                target_languages, 
                context, 
                model, 
                translation_mode
            )
            
            # Ensure the source language is in the results
            if source_language not in translations_results:
                translations_results[source_language] = text_to_translate.strip('"')
        
        try:
            # Get Firebase reference for the group message
            messages_ref = db.reference(f'group_messages/{group_id}/{message_id}')
            
            # Use the original message in the source language for the message field
            original_message = text_to_translate.strip('"')
            
            # Check if this is a new message
            message_data = messages_ref.get()
            
            # Only set the message field for new messages
            if message_data is None:
                # This is a new message, set the message field
                messages_ref.update({
                    'message': original_message,
                    'senderLanguage': source_language,  # Only use senderLanguage
                    'translationMode': translation_mode,
                })
            else:
                # This is an existing message, only update metadata
                messages_ref.update({
                    'senderLanguage': source_language,  # Only use senderLanguage
                    'translationMode': translation_mode,
                })
            
            # Remove source language from translations since it's redundant with message field
            if source_language in translations_results:
                del translations_results[source_language]
                
            # Update the translations map
            translations_ref = messages_ref.child('translations')
            translations_ref.update(translations_results)
        except Exception as firebase_error:
            print(f"Error updating Firebase: {str(firebase_error)}")
            return Response({"error": f"Firebase update failed: {str(firebase_error)}"}, status=500)
        
        return Response({
            'status': 'success',
            'translations_count': len(translations_results),
            'languages': list(translations_results.keys()),
            'original_text': text_to_translate,
            'translation_mode': translation_mode,
            'context_used': len(context) > 0,
            'context_messages': len(context),
            'batch_translation': True
        })
        
    except ValueError as e:
        print(f"Value error in context translation: {str(e)}")
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        print(f"Context-aware translation failed: {str(e)}")
        return Response({"error": f"Context-aware translation failed: {str(e)}"}, status=500)

# ==============================================
# Admin API Views
# ==============================================

# NOTE: These are placeholders. Authentication, permissions, and actual logic need implementation.

@api_view(['POST'])
@permission_classes([AllowAny]) # Login endpoint itself doesn't require authentication
def admin_login(request):
    """
    Admin login endpoint.
    Authenticates against Django users and returns an API token.
    """
    username = request.data.get('email') # Assuming email is used as username
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'Please provide both email and password'}, status=status.HTTP_400_BAD_REQUEST)

    # Authenticate user using Django's auth system
    # Note: Ensure your AUTHENTICATION_BACKENDS support email/username login if needed
    user = authenticate(request=request, username=username, password=password)

    if user is not None:
        # Check if the user is an admin (staff)
        if user.is_staff:
            # Get or create a token for the user
            token, created = Token.objects.get_or_create(user=user)
            return Response({'token': token.key})
        else:
            # User authenticated but is not an admin
            return Response({'error': 'User is not authorized for admin access'}, status=status.HTTP_403_FORBIDDEN)
    else:
        # Authentication failed
        return Response({'error': 'Invalid Credentials'}, status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_logout(request):
    """
    Admin logout endpoint.
    Needs logic to invalidate token/session.
    """
    # For TokenAuthentication, logout is typically handled client-side by deleting the token.
    # If you need server-side token invalidation, you can delete the token:
    try:
        # request.auth is the token object provided by TokenAuthentication
        if request.auth:
            request.auth.delete()
            return Response({'message': 'Logged out successfully by deleting token.'})
        else:
             # This might happen if the request didn't include a valid token
             return Response({'message': 'Logout successful (no token found to delete).'}) 
    except Exception as e:
        print(f"Error during logout: {e}")
        return Response({'error': 'An error occurred during logout.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser]) # Restore IsAdminUser permission check
def admin_user_list_create(request):
    """
    GET: List all users.
    POST: Create a new user.
    Needs implementation using Firebase Admin SDK.
    """
    if request.method == 'GET':
        try:
            users_ref = db.reference('users')
            snapshot = users_ref.get() # Use get() instead of once('value') in Python SDK
            # Firebase returns None if the path doesn't exist or has no data
            if snapshot is None:
                return Response({}) # Return empty dict if no users
            return Response(snapshot)
        except Exception as e:
            print(f"Error fetching users from Firebase: {e}")
            return Response({'error': f'Failed to fetch users: {str(e)}'}, status=500)

    elif request.method == 'POST':
        try:
            email = request.data.get('email')
            password = request.data.get('password')

            if not email or not password:
                return Response({'error': 'Email and password are required'}, status=400)

            # Create the user in Firebase Authentication
            user_record = auth.create_user(
                email=email,
                password=password,
                email_verified=False
            )

            # Create the user in Realtime Database
            user_ref = db.reference(f'users/{user_record.uid}')
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            user_data = {
                'userId': user_record.uid,
                'email': email,
                'accountType': 'free',
                'username': email.split('@')[0],
                'status': 'offline',
                'language': 'English', # Default language
                'translator': 'google', # Default translator
                'createdAt': now,
                'lastLoginDate': now
            }
            user_ref.set(user_data)

            return Response({'message': 'User created successfully', 'userId': user_record.uid}, status=201)
        
        except auth.EmailAlreadyExistsError:
             return Response({'error': 'Email already exists'}, status=400)
        except Exception as e:
            print(f"Error creating user: {e}")
            # Attempt to clean up if user was created in Auth but failed before RTDB
            if 'user_record' in locals() and user_record:
                try:
                    auth.delete_user(user_record.uid)
                    print(f"Cleaned up partially created user: {user_record.uid}")
                except Exception as cleanup_error:
                    print(f"Failed to cleanup partially created user {user_record.uid}: {cleanup_error}")
            return Response({'error': f'Failed to create user: {str(e)}'}, status=500)

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_user_detail_update_delete(request, user_id):
    """
    GET: Retrieve user details.
    PUT: Update user details.
    DELETE: Delete user.
    Needs implementation using Firebase Admin SDK.
    """
    user_ref = db.reference(f'users/{user_id}')

    if request.method == 'GET':
        try:
            snapshot = user_ref.get()
            if snapshot is None:
                return Response({'error': 'User not found'}, status=404)
            return Response(snapshot)
        except Exception as e:
            print(f"Error fetching user {user_id}: {e}")
            return Response({'error': f'Failed to fetch user: {str(e)}'}, status=500)

    elif request.method == 'PUT':
        try:
            # Check if user exists in RTDB before attempting update
            if user_ref.get() is None:
                 return Response({'error': 'User not found in Realtime Database'}, status=404)

            user_data = request.data
            # Ensure userId is not part of the update payload
            if 'userId' in user_data:
                del user_data['userId'] 
            if 'email' in user_data:
                 # Updating email in RTDB might be okay, but Auth email is separate
                 # Consider if you need to update Auth email as well (requires re-auth typically)
                 # For now, we only update RTDB as per Node.js logic.
                 pass 
            
            # Prevent overwriting critical fields if not provided
            # Ensure essential keys like createdAt are not accidentally removed if not in request.data
            # It might be safer to fetch existing data and merge updates
            # However, mirroring Node.js simple update for now:
            user_ref.update(user_data)
            
            return Response({'message': 'User updated successfully'})
        except Exception as e:
            print(f"Error updating user {user_id}: {e}")
            return Response({'error': f'Failed to update user: {str(e)}'}, status=500)

    elif request.method == 'DELETE':
        try:
            # Check if user exists in RTDB before attempting delete
            if user_ref.get() is None:
                print(f"User {user_id} not found in Realtime Database, attempting Auth delete only.")
                # Proceed to delete from Auth even if not in RTDB
            else:
                # Delete user from Realtime Database first
                 user_ref.delete()
                 print(f"Deleted user {user_id} from Realtime Database.")

            # Delete user from Authentication
            try:
                auth.delete_user(user_id)
                print(f"Deleted user {user_id} from Firebase Authentication.")
            except auth.UserNotFoundError:
                 print(f"User {user_id} not found in Firebase Authentication (already deleted or never existed?).")
                 # If user wasn't in RTDB either, return 404. Otherwise, it's a partial success.
                 if user_ref.get() is None: # Check RTDB again in case of race condition? No, just use initial check. Assume RTDB delete worked if attempted.
                     return Response({'error': 'User not found in RTDB or Auth'}, status=404)
                 else: # User was deleted from RTDB but not found in Auth
                    return Response({'message': 'User deleted from Realtime Database, but not found in Authentication.'}) 

            return Response({'message': 'User deleted successfully'})
        except Exception as e:
            print(f"Error deleting user {user_id}: {e}")
            # Consider potential partial deletion scenarios
            return Response({'error': f'Failed to delete user: {str(e)}'}, status=500)

@api_view(['GET'])
@permission_classes([IsAdminUser]) # Re-enabled permission check
def admin_usage_stats(request):
    """
    Get usage statistics (placeholder).
    Needs implementation using Firebase Admin SDK.
    """
    # TODO: Implement usage stats calculation based on Firebase RTDB data
    try:
        users_ref = db.reference('users')
        users_snapshot = users_ref.get()
        users = users_snapshot if users_snapshot else {}

        daily_login_usage = []
        today = datetime.date.today()

        # Calculate logins for the last 7 days (today + 6 previous days)
        for i in range(6, -1, -1):
            target_date = today - datetime.timedelta(days=i)
            date_str = target_date.isoformat() # YYYY-MM-DD format
            
            users_logged_in_on_date = []
            count = 0
            for user_id, user_data in users.items():
                last_login_iso = user_data.get('lastLoginDate')
                if last_login_iso:
                    try:
                        # Extract date part from ISO string (e.g., "2024-07-28T10:00:00Z")
                        last_login_date_str = last_login_iso.split('T')[0]
                        if last_login_date_str == date_str:
                            count += 1
                            users_logged_in_on_date.append({
                                'userId': user_id, # Include userId for potential UI use
                                'email': user_data.get('email'),
                                'accountType': user_data.get('accountType'),
                                'loginTime': last_login_iso
                            })
                    except Exception as date_parse_error:
                        # Log if a date string is invalid, but continue
                        print(f"Could not parse lastLoginDate '{last_login_iso}' for user {user_id}: {date_parse_error}")
                        continue 

            daily_login_usage.append({
                'date': date_str,
                'count': count,
                'users': users_logged_in_on_date
            })

        return Response({
            'dailyLoginUsage': daily_login_usage
        })

    except Exception as e:
        print(f'Error fetching usage statistics: {e}')
        return Response({'error': 'Failed to fetch usage statistics'}, status=500)
