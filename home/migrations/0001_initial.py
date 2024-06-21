# Generated by Django 5.0.1 on 2024-04-23 05:46

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Messages",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user1", models.TextField(default="")),
                ("user2", models.CharField(default="", max_length=1000)),
            ],
        ),
    ]