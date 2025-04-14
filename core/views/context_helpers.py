from firebase_admin import db
from google.genai import types
import anthropic
import os
from openai import OpenAI

# Import helpers from the new location
from .translation_helpers import get_translation_instruction, get_translation, anthropic_client, deepseek_client, gemini_clients

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