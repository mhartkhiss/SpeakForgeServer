from django.db import models

# Create your models here.

class TranslationCache(models.Model):
    source_text = models.TextField()
    source_language = models.CharField(max_length=50)
    target_language = models.CharField(max_length=50)
    translated_text = models.TextField()
    model = models.CharField(max_length=20)  # claude, gemini, or deepseek
    mode = models.CharField(max_length=10)  # single or multiple
    created_at = models.DateTimeField(auto_now_add=True)
    used_count = models.IntegerField(default=1)
    
    class Meta:
        indexes = [
            models.Index(fields=['source_text', 'source_language', 'target_language', 'model', 'mode']),
        ]

    def __str__(self):
        return f"{self.source_language} -> {self.target_language}: {self.source_text[:50]}"

# Add new model for translation memory
class GroupTranslationMemory(models.Model):
    group_id = models.CharField(max_length=100)
    original_term = models.CharField(max_length=255)
    translated_term = models.CharField(max_length=255)
    source_language = models.CharField(max_length=10)
    target_language = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    usage_count = models.IntegerField(default=1)
    
    class Meta:
        unique_together = ('group_id', 'original_term', 'source_language', 'target_language')
        indexes = [
            models.Index(fields=['group_id']),
            models.Index(fields=['original_term']),
        ]
        verbose_name_plural = 'Group Translation Memories'
        
    def __str__(self):
        return f"{self.group_id}: {self.original_term[:30]} -> {self.translated_term[:30]}"
