from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.reverse import reverse

@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request, format=None):
    """
    API root endpoint that lists all available endpoints.
    """
    return Response({
        'status': 'Welcome to SpeakForge API',
        'version': '1.0.0',
        'endpoints': {
            'translate': reverse('core:translate', request=request, format=format),
            # Add other endpoints here as needed
        }
    }) 