from ..models import GroupTranslationMemory

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