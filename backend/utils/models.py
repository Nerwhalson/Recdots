from django.db import models

# Create your models here.

class User(models.Model):
	# type字段
    user_id = models.IntegerField(primary_key=True, db_index=True)
    user_name = models.CharField(max_length=255)
    user_type = models.CharField(max_length=255)
    gender = models.CharField(max_length=8, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    create_time = models.DateTimeField()
    update_time = models.DateTimeField()


class Item(models.Model):
    item_id = models.IntegerField(primary_key=True, db_index=True)
    item_name = models.CharField(max_length=255)
    item_type = models.CharField(max_length=255)
    clicked_time = models.IntegerField(default=0)
    create_time = models.DateTimeField()
    update_time = models.DateTimeField()


class Behavior(models.Model):
	# create_time和update_time
    index = models.BigAutoField(primary_key=True, db_index=True)
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    create_time = models.DateTimeField()
    update_time = models.DateTimeField()
    behave_type = models.CharField(max_length=255)
    score = models.IntegerField(null=True, blank=True)
    

class Rec(models.Model):
    index = models.BigAutoField(primary_key=True, db_index=True)
    user_id = models.IntegerField()
    item_ids = models.CharField(max_length=255)
    clicked_ids = models.CharField(max_length=1024, null=True, blank=True)
    create_time = models.DateTimeField()
    update_time = models.DateTimeField()