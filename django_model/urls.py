from django.urls import path,include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('django_app.urls')),
    path('prediction', views.prediction, name='prediction'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)