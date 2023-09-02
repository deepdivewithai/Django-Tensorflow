from rest_framework import serializers
from .models import Image


class ImagesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'