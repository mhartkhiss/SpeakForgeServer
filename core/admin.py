from django.contrib import admin
from .models import TranslationCache

@admin.register(TranslationCache)
class TranslationCacheAdmin(admin.ModelAdmin):
    list_display = ('source_language', 'target_language', 'model', 'mode', 'used_count', 'created_at')
    list_filter = ('source_language', 'target_language', 'model', 'mode')
    search_fields = ('source_text', 'translated_text')
    readonly_fields = ('created_at', 'used_count')
