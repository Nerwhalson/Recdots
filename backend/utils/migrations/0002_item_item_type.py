# Generated by Django 4.1 on 2022-08-28 09:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("utils", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="item",
            name="item_type",
            field=models.CharField(default=None, max_length=255),
            preserve_default=False,
        ),
    ]