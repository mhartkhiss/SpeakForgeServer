import anthropic
import os
import json
import requests
from openai import OpenAI
from google import genai
from google.genai import types
from firebase_admin import db

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
            "Format: 1. [translation]\n2. [translation]\n3. [translation]"
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