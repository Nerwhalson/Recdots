from django.db import models

# Create your models here.

class User(models.Model):
	# type字段
    user_id = models.IntegerField()
    user_name = models.CharField(max_length=255)
    gender = models.CharField(max_length=255)
    age = models.IntegerField()
    create_date = models.DateField()


class Item(models.Model):
    item_id = models.IntegerField()
    item_name = models.CharField(max_length=255)
    create_date = models.DateField()


class Behavior(models.Model):
	# create_time和update_time
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    behave_time = models.TimeField()
    behave_type = models.CharField(max_length=255)
    score = models.IntegerField()
    

class Rec(models.Model):
	# item_ids改成用一个更长的char字段；自增主键；clicked_ids字段，记录点击
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    rec_time = models.TimeField()