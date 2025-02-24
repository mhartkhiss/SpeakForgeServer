from django.apps import AppConfig
from .firebase_init import initialize_firebase


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        initialize_firebase()
