from flask import Flask, request, jsonify, Response
import requests
import json
import os

app = Flask(__name__)

# Configuración
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', 'tu_api_key_aqui')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # Obtener datos de la solicitud
        data = request.get_json()
        
        # Mapear el modelo si es necesario
        model_mapping = {
            'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
            'gpt-4': 'meta/llama-3.1-70b-instruct',
            'gpt-4-turbo': 'meta/llama-3.1-70b-instruct'
        }
        
        # Reemplazar modelo si existe en el mapeo
        if data.get('model') in model_mapping:
            data['model'] = model_mapping[data['model']]
        
        # Preparar headers para Nvidia
        headers = {
            'Authorization': f'Bearer {NVIDIA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Hacer solicitud a Nvidia NIM
        nvidia_response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            stream=data.get('stream', False)
        )
        
        # Si es streaming, devolver como stream
        if data.get('stream', False):
            def generate():
                for chunk in nvidia_response.iter_lines():
                    if chunk:
                        yield chunk + b'\n'
            
            return Response(generate(), content_type='text/event-stream')
        
        # Respuesta normal
        return jsonify(nvidia_response.json()), nvidia_response.status_code
        
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'proxy_error'
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Endpoint para listar modelos disponibles"""
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'gpt-3.5-turbo',
                'object': 'model',
                'created': 1677610602,
                'owned_by': 'openai-proxy'
            },
            {
                'id': 'gpt-4',
                'object': 'model',
                'created': 1687882411,
                'owned_by': 'openai-proxy'
            }
        ]
    })

@app.route('/', methods=['GET'])
def home():
    """Endpoint raíz"""
    return jsonify({
        'status': 'running',
        'message': 'OpenAI to Nvidia NIM Proxy',
        'endpoints': ['/v1/chat/completions', '/v1/models', '/health']
    })

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
