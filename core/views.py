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
from .models import TranslationCache, GroupTranslationMemory
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

def update_firebase_message(ref_path, room_id, message_id, translations, source_language, translation_mode, is_group=False, target_language=None):
    """
    Helper function to update Firebase with translation data
    """
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
        # For direct messages
        if 'var1' in translations:
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
        # Get translation
        translated_text = get_translation(text_to_translate, source_language, target_language, variants, translation_mode, model)

        # Process translations
        translations = process_translations(translated_text, variants)
        
        # Update Firebase if message_id and room_id provided
        if message_id and room_id:
            # Determine the correct Firebase reference path based on is_group flag
            ref_path = 'group_messages' if is_group else 'messages'
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
        return Response({"error": str(e)}, status=400)
    except Exception as e:
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
    
    # Format context for AI models
    formatted_context = []
    for msg in context_messages:
        msg_data = msg['data']
        
        # Use original message if available, otherwise use translated message
        message_text = msg_data.get('messageOG', msg_data.get('message', ''))
        
        # If source language is specified and we have translations, use the specific language
        if source_language and 'translations' in msg_data and source_language in msg_data['translations']:
            message_text = msg_data['translations'][source_language]
            
        sender_name = msg_data.get('senderName', 'Unknown User')
        formatted_context.append(f"{sender_name}: {message_text}")
    
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
            
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.1,
                system=system_message,
                messages=[{"role": "user", "content": text}]
            )
            
            return message.content[0].text.strip() if message.content else "Translation failed."
            
        elif model == 'gemini':
            # Configure generation parameters
            generation_config = types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                system_instruction=context_instruction
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
                        {"role": "system", "content": context_instruction},
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

@api_view(['POST'])
@permission_classes([AllowAny])
def translate_group_context(request):
    """
    Enhanced endpoint for translating group messages with conversation context.
    This endpoint handles:
    1. Fetching previous messages as context
    2. Translating with context awareness 
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
        
        # Results dictionary
        translations_results = {}
        
        # Process each target language
        for target_language in target_languages:
            # Skip if source and target are the same
            if source_language == target_language or (source_language == 'auto' and target_language == 'en'):
                translations_results[target_language] = text_to_translate.strip('"')
                continue
                
            try:
                # Get context-aware translation
                translated_text = translate_with_context(
                    text_to_translate, 
                    source_language, 
                    target_language, 
                    context, 
                    model, 
                    translation_mode
                )
                
                # Process translation result
                translations_results[target_language] = translated_text.strip()
                
                # Update translation memory
                try:
                    if translated_text:
                        update_translation_memory(
                            group_id, 
                            text_to_translate, 
                            translated_text, 
                            source_language, 
                            target_language
                        )
                except Exception as memory_error:
                    print(f"Error updating translation memory: {str(memory_error)}")
                    # Continue even if memory update fails
            except Exception as translate_error:
                print(f"Error translating to {target_language}: {str(translate_error)}")
                # Use original text if translation fails
                translations_results[target_language] = text_to_translate.strip('"')
        
        # Also add the original language to translations
        translations_results[source_language] = text_to_translate.strip('"')
        
        try:
            # Get Firebase reference for the group message
            messages_ref = db.reference(f'group_messages/{group_id}/{message_id}')
            
            # Set default translated message (use first translation or original if no translations)
            default_translation = next(iter(translations_results.values())) if translations_results else text_to_translate.strip('"')
            
            # Update the message with translations
            messages_ref.update({
                'message': default_translation,
                'messageOG': text_to_translate.strip('"'),
                'sourceLanguage': source_language,
                'translationMode': translation_mode,
            })
            
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
            'context_messages': len(context)
        })
        
    except ValueError as e:
        print(f"Value error in context translation: {str(e)}")
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        print(f"Context-aware translation failed: {str(e)}")
        return Response({"error": f"Context-aware translation failed: {str(e)}"}, status=500)
