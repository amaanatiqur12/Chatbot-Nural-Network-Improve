# Generated by Django 5.0.1 on 2024-05-29 16:11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("home", "0004_messages_timestamp"),
    ]

    operations = [
        migrations.AddField(
            model_name="messages",
            name="sentiment_analysis",
            field=models.CharField(default="", max_length=10),
        ),
    ]