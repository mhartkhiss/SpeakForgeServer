from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from firebase_admin import db

# Import helpers from the new location
from .translation_helpers import get_translation, process_translations, update_firebase_message
from .context_helpers import get_conversation_context, translate_with_batch_context # Need translate_group_context endpoint here

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