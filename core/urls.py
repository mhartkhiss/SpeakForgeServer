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

    # Admin API Endpoints (No longer prefixed with /api/)
    path('admin/login/', views.admin_login, name='admin_login'),
    path('admin/logout/', views.admin_logout, name='admin_logout'),
    path('admin/users/', views.admin_user_list_create, name='admin_user_list_create'),
    path('admin/users/<str:user_id>/', views.admin_user_detail_update_delete, name='admin_user_detail_update_delete'),
    # path('admin/reset-password/', views.admin_reset_password, name='admin_reset_password'), # Handled frontend now
    path('admin/usage/', views.admin_usage_stats, name='admin_usage_stats'),
] 