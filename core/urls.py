from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.api_root, name='api_root'),
    path('translate/', views.translate, name='translate'),
    path('translate-db/', views.translate_db, name='translate_db'),
    path('translate-batch/', views.translate_batch, name='translate_batch'),
    path('translate-group/', views.translate_group, name='translate_group'),
    path('translate-group-context/', views.translate_group_context, name='translate_group_context'),
    path('regenerate-translation/', views.regenerate_translation, name='regenerate_translation'),
    path('translator/', views.TranslatorView.as_view(), name='translator-interface'),
] 