from django.db import models

class Image(models.Model):
    images = models.ImageField('./media')

    def __str__(self) -> str:
        return str(self.pk)