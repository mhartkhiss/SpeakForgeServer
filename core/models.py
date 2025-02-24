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
