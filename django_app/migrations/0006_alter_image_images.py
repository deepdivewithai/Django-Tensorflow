# Generated by Django 4.2.4 on 2023-09-02 07:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_app', '0005_alter_image_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='images',
            field=models.ImageField(blank=True, null=True, upload_to='pictures'),
        ),
    ]
