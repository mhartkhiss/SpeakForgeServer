from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.api_root, name='api-root'),
    path('translate/', views.translate, name='translate'),
    path('translate-db/', views.translate_db, name='translate-db'),
    path('translator/', views.TranslatorView.as_view(), name='translator-interface'),
] 