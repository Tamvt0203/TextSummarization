from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .apps import TsAppConfig

@csrf_exempt
def summarize(request):
    if request.method == 'POST':
        try:
            # Phân tích JSON từ request body
            data = json.loads(request.body)
            print(data)
            text = data.get('text', '')
            language = data.get('language', 'en')  # Mặc định là 'en' nếu không được cung cấp

            if not text:
                return JsonResponse({'error': 'No text provided for summarization'}, status=400)

            if language not in ['en', 'vi']:
                return JsonResponse({'error': 'Unsupported language'}, status=400)

            # Truy cập mô hình từ AppConfig
            model_loader = TsAppConfig.model_loader
            summary = model_loader.summarize_text(text, language=language)
            return JsonResponse({'summary': summary})
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'This endpoint supports only POST requests'}, status=405)
