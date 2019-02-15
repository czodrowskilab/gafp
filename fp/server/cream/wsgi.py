"""
WSGI config for cream project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

from cream.trained_models import model_manager

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cream.settings')

model_manager.load_all_models()
application = get_wsgi_application()
