from django.apps import AppConfig
from .models import ModelLoader
import logging

class TsAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ts_app'
    # Khởi tạo biến class để lưu trữ model_loader
    model_loader =  ModelLoader(
    model_checkpoints={'en': 't5-base', 'vi': 'VietAI/vit5-base-vietnews-summarization'},
    weights_files={'en': 'ts_app/models/my_model_en_weights.pth', 'vi': 'ts_app/models/my_model_vi_weights.pth'}
)

